#!/usr/bin/env python3
"""
Fine-tune bge-reranker-v2-m3 with DoRA (or LoRA fallback).
Creates a domain-adapted reranker for recruiting ranking.
Adapter is CPU-compatible and can be loaded on CPU.
"""

import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import argparse
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Try to import peft
try:
    from peft import get_peft_model, LoraConfig, TaskType
    HAS_PEFT = True
    print("✓ PEFT library available")
except ImportError:
    HAS_PEFT = False
    print("✗ PEFT library not found. Install: pip install peft")

# Check if DoRA is available (not in all PEFT versions)
SUPPORTS_DORA = False
if HAS_PEFT:
    try:
        from peft import DoraConfig
        SUPPORTS_DORA = True
        print("✓ DoRA support available")
    except ImportError:
        print("ℹ DoRA not available in this PEFT version, using LoRA (equally excellent)")


class RankingPairDataset(Dataset):
    """Dataset of (JD, candidate, label) pairs for reranker fine-tuning."""
    
    def __init__(self, data_file, tokenizer, max_length=512):
        """Load and preprocess pairs."""
        with open(data_file) as f:
            data = json.load(f)

        # Accept both {"pairs": [...]} and flat list formats
        if isinstance(data, list):
            self.pairs = data
        else:
            self.pairs = data['pairs']
        self.tokenizer = tokenizer
        self.max_length = max_length

        print(f"Loaded {len(self.pairs)} pairs from {data_file}")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Combine JD and candidate for reranker
        # Reranker expects two sentences: [JD | candidate]
        jd = pair['jd']
        candidate = pair['candidate']
        label = pair['label']
        
        # Tokenize pair
        encoded = self.tokenizer(
            jd,
            candidate,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.float),
        }


def create_model_and_adapter(model_name, use_dora=True, rank=8, alpha=16):
    """
    Initialize reranker model with DoRA or LoRA adapter.
    
    Args:
        model_name: HuggingFace model ID
        use_dora: Use DoRA if available, else LoRA
        rank: LoRA rank (dimension of updates)
        alpha: LoRA scaling factor
    """
    print(f"\nLoading base model: {model_name}")
    
    # num_labels=1 matches the pretrained bge-reranker-v2-m3 classifier shape.
    # Using num_labels=2 reinitializes the pretrained [1,1024] classifier to random
    # [2,1024] weights, losing all pretrained relevance signal. We keep num_labels=1
    # and use BCEWithLogitsLoss manually for binary fine-tuning.
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        device_map='auto' if torch.cuda.is_available() else None,
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    if not HAS_PEFT:
        print("PEFT not available - returning base model only")
        return model, None
    
    # Choose adapter type
    adapter_type = 'DoRA' if (use_dora and SUPPORTS_DORA) else 'LoRA'
    print(f"Creating {adapter_type} adapter (rank={rank}, alpha={alpha})")
    
    if use_dora and SUPPORTS_DORA:
        try:
            config = DoraConfig(
                r=rank,
                lora_alpha=alpha,
                lora_dropout=0.1,
                bias='none',
                target_modules=['query', 'value', 'key', 'dense'],  # Cross-encoder projections
                task_type=TaskType.SEQ_CLS,
            )
        except Exception as e:
            print(f"DoRA config failed: {e}, falling back to LoRA")
            config = LoraConfig(
                r=rank,
                lora_alpha=alpha,
                lora_dropout=0.1,
                bias='none',
                target_modules=['query', 'value', 'key', 'dense'],
                task_type=TaskType.SEQ_CLS,
            )
    else:
        config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0.1,
            bias='none',
            target_modules=['query', 'value', 'key', 'dense'],
            task_type=TaskType.SEQ_CLS,
        )
    
    # Apply adapter
    model = get_peft_model(model, config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    print(f"Adapter applied:")
    print(f"  Total params: {total:,}")
    print(f"  Trainable params: {trainable:,}")
    print(f"  Trainable %: {100 * trainable / total:.2f}%")
    
    return model, config


def train_epoch(model, dataloader, optimizer, device, scaler=None, grad_accum=1):
    """Train one epoch with optional fp16 and gradient accumulation."""
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    progress_bar = tqdm(dataloader, desc='Training')
    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        bce_loss = torch.nn.BCEWithLogitsLoss()
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = bce_loss(outputs.logits.squeeze(-1), labels) / grad_accum
            scaler.scale(loss).backward()
            if (step + 1) % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = bce_loss(outputs.logits.squeeze(-1), labels) / grad_accum
            loss.backward()
            if (step + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item() * grad_accum
        progress_bar.set_postfix({'loss': loss.item() * grad_accum})

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            bce_loss = torch.nn.BCEWithLogitsLoss()
            logits = outputs.logits.squeeze(-1)
            loss = bce_loss(logits, labels)

            total_loss += loss.item()

            predictions = (logits.sigmoid() > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(dataloader), correct / total


def train_model(
    model,
    train_dataloader,
    val_dataloader,
    num_epochs=3,
    learning_rate=2e-4,
    output_dir='reranker_adapter',
    device='cuda',
    fp16=True,
    grad_accum=4,
):
    """Fine-tune the model with optional fp16 and gradient accumulation."""
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler('cuda') if (fp16 and device == 'cuda') else None

    print(f"\nTraining for {num_epochs} epochs on {device}")
    print(f"Learning rate: {learning_rate}, fp16={fp16}, grad_accum={grad_accum}")
    print(f"Output dir: {output_dir}\n")

    Path(output_dir).mkdir(exist_ok=True)

    best_val_loss = float('inf')
    patience = 2
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")

        # Train
        train_loss = train_epoch(model, train_dataloader, optimizer, device, scaler=scaler, grad_accum=grad_accum)
        print(f"Train loss: {train_loss:.4f}")
        
        # Evaluate
        if val_dataloader:
            val_loss, val_acc = evaluate(model, val_dataloader, device)
            print(f"Val loss: {val_loss:.4f}, Val accuracy: {val_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                model.save_pretrained(f"{output_dir}/best")
                print(f"✓ Saved best model to {output_dir}/best")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
    
    # Save final adapter
    model.save_pretrained(output_dir)
    print(f"\n✓ Final adapter saved to {output_dir}")
    
    return model


def load_adapter_cpu(model_name, adapter_path):
    """Load model and adapter on CPU for inference."""
    print(f"Loading model {model_name}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
    )
    
    if HAS_PEFT:
        from peft import PeftModel
        print(f"Loading adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Fine-tune reranker with DoRA/LoRA')
    parser.add_argument('--data', type=str, default='synthetic_training_data.json',
                       help='Training data file')
    parser.add_argument('--model', type=str, default='BAAI/bge-reranker-v2-m3',
                       help='Base model')
    parser.add_argument('--use-dora', action='store_true', default=True,
                       help='Use DoRA if available')
    parser.add_argument('--rank', type=int, default=8,
                       help='LoRA rank')
    parser.add_argument('--alpha', type=int, default=16,
                       help='LoRA alpha')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--output', type=str, default='reranker_adapter',
                       help='Output directory')
    parser.add_argument('--val-split', type=float, default=0.1,
                       help='Validation split')
    args = parser.parse_args()
    
    # Check data file exists
    if not Path(args.data).exists():
        print(f"Data file not found: {args.data}")
        print("Generate it first with: python generate_synthetic_data.py")
        return
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Use pre-assigned splits if 'split' field is present, else random split
    with open(args.data) as f:
        raw = json.load(f)
    pairs = raw if isinstance(raw, list) else raw['pairs']
    has_splits = isinstance(pairs[0], dict) and 'split' in pairs[0]

    if has_splits:
        import tempfile, os
        train_pairs = [p for p in pairs if p.get('split') != 'val']
        val_pairs   = [p for p in pairs if p.get('split') == 'val']
        # Write temp files for each split
        _train_tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(train_pairs, _train_tmp); _train_tmp.close()
        _val_tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(val_pairs, _val_tmp); _val_tmp.close()
        train_dataset = RankingPairDataset(_train_tmp.name, tokenizer)
        val_dataset   = RankingPairDataset(_val_tmp.name, tokenizer)
        os.unlink(_train_tmp.name); os.unlink(_val_tmp.name)
    else:
        dataset = RankingPairDataset(args.data, tokenizer)
        val_size = int(len(dataset) * args.val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
    ) if len(val_dataset) > 0 else None

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model and adapter
    model, config = create_model_and_adapter(
        args.model,
        use_dora=args.use_dora,
        rank=args.rank,
        alpha=args.alpha,
    )
    
    model.to(device)
    
    # Train
    if HAS_PEFT:
        model = train_model(
            model,
            train_dataloader,
            val_dataloader,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            output_dir=args.output,
            device=device,
        )
        
        print(f"\n✅ Fine-tuning complete!")
        print(f"Adapter saved to: {args.output}")
        print(f"\nTo use on CPU for inference:")
        print(f"  from transformers import AutoTokenizer, AutoModelForSequenceClassification")
        print(f"  from peft import PeftModel")
        print(f"  ")
        print(f"  model = AutoModelForSequenceClassification.from_pretrained('{args.model}')")
        print(f"  model = PeftModel.from_pretrained(model, '{args.output}')")
        print(f"  tokenizer = AutoTokenizer.from_pretrained('{args.model}')")
    else:
        print("PEFT not available - model cannot be fine-tuned")


if __name__ == '__main__':
    main()
