#!/usr/bin/env python3
"""
CPU-friendly reranker using DoRA/LoRA adapter.
Takes semantic top-500 and re-ranks with fine-tuned model.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import argparse
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Optional PEFT for loading adapters
try:
    from peft import PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False


class RerankerInferenceDataset(Dataset):
    """Dataset for batch reranking."""
    
    def __init__(self, jd_text, candidates_list, tokenizer, max_length=512):
        """
        Args:
            jd_text: Job description
            candidates_list: List of candidate texts
            tokenizer: Tokenizer
            max_length: Max sequence length
        """
        self.jd_text = jd_text
        self.candidates = candidates_list
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.candidates)
    
    def __getitem__(self, idx):
        candidate = self.candidates[idx]
        
        encoded = self.tokenizer(
            self.jd_text,
            candidate,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'index': idx,
        }


class CPUReranker:
    """CPU-friendly reranker using fine-tuned adapter."""
    
    def __init__(self, model_name='BAAI/bge-reranker-v2-m3', adapter_path=None):
        """
        Initialize reranker.
        
        Args:
            model_name: Base model ID
            adapter_path: Path to LoRA/DoRA adapter (optional)
        """
        print(f"Loading reranker model: {model_name}")
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            ignore_mismatched_sizes=True,
            torch_dtype=torch.float32,  # Use float32 for CPU
        )
        
        # Load adapter if provided
        if adapter_path and HAS_PEFT:
            print(f"Loading adapter from: {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.has_adapter = True
        else:
            self.has_adapter = False
        
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Move to CPU (no GPU needed!)
        self.model.cpu()
        
        print(f"Model ready on CPU (parameters: {sum(p.numel() for p in self.model.parameters()):,})")
    
    def rerank(self, jd_text, candidate_texts, batch_size=32, top_k=None):
        """
        Rerank candidates for a JD.
        
        Args:
            jd_text: Job description
            candidate_texts: List of candidate texts
            batch_size: Batch size for inference
            top_k: Return only top-k (None = return all)
        
        Returns:
            List of (index, score) tuples, sorted by score descending
        """
        dataset = RerankerInferenceDataset(
            jd_text,
            candidate_texts,
            self.tokenizer,
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        
        scores = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Reranking', disable=len(candidate_texts) < 100):
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                indices = batch['index']
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                
                logits = outputs.logits
                # Score is softmax over class 1 (positive relevance)
                batch_scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                
                for idx, score in zip(indices, batch_scores):
                    scores.append((idx.item(), float(score)))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        if top_k:
            scores = scores[:top_k]
        
        return scores
    
    def rerank_dataframe(self, jd_text, candidates_df, candidate_col='text', top_k=100):
        """
        Rerank a dataframe of candidates.
        
        Args:
            jd_text: Job description
            candidates_df: DataFrame with candidates
            candidate_col: Column name with candidate text
            top_k: Return only top-k
        
        Returns:
            DataFrame with reranking scores
        """
        candidate_texts = candidates_df[candidate_col].tolist()
        
        scores = self.rerank(jd_text, candidate_texts, top_k=None)
        
        # Build result dataframe
        result_rows = []
        for rank, (idx, score) in enumerate(scores[:top_k], 1):
            row = candidates_df.iloc[idx].copy()
            row['rerank_score'] = score
            row['rerank_position'] = rank
            result_rows.append(row)
        
        return pd.DataFrame(result_rows)


def main():
    parser = argparse.ArgumentParser(description='Rerank with fine-tuned model on CPU')
    parser.add_argument('--model', type=str, default='BAAI/bge-reranker-v2-m3',
                       help='Base model')
    parser.add_argument('--adapter', type=str, default=None,
                       help='Path to LoRA/DoRA adapter')
    parser.add_argument('--input-csv', type=str, default='enhanced_top100.csv',
                       help='Input CSV with candidates to rerank')
    parser.add_argument('--jd-text', type=str, default=None,
                       help='JD text')
    parser.add_argument('--jd-file', type=str, default=None,
                       help='JD file')
    parser.add_argument('--output', type=str, default='reranked_results.csv',
                       help='Output file')
    parser.add_argument('--top-k', type=int, default=100,
                       help='Return top-k candidates')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for reranking')
    args = parser.parse_args()
    
    # Load JD
    if args.jd_file:
        with open(args.jd_file) as f:
            jd_text = f.read()
    elif args.jd_text:
        jd_text = args.jd_text
    else:
        jd_text = """
        Senior AI Engineer
        5+ years of experience in Machine Learning and Deep Learning.
        Required: Python, PyTorch, Transformers, NLP, LLMs.
        Preferred: Distributed training, LoRA fine-tuning, MLOps.
        """
    
    # Load candidates
    print(f"Loading candidates from {args.input_csv}...")
    candidates_df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(candidates_df)} candidates")
    
    # Create reranker (CPU-only)
    reranker = CPUReranker(
        model_name=args.model,
        adapter_path=args.adapter,
    )
    
    # Prepare candidate texts
    # Combine relevant columns into text for reranking
    def make_candidate_text(row):
        parts = []
        if 'name' in row and pd.notna(row.get('name')):
            parts.append(f"Name: {row['name']}")
        if 'location' in row and pd.notna(row.get('location')):
            parts.append(f"Location: {row['location']}")
        parts.append(f"Score: {row.get('score', 0):.2f}")
        return " | ".join(parts)
    
    candidates_df['text'] = candidates_df.apply(make_candidate_text, axis=1)
    
    # Rerank
    print(f"\nReranking {len(candidates_df)} candidates...")
    results_df = reranker.rerank_dataframe(
        jd_text,
        candidates_df,
        candidate_col='text',
        top_k=args.top_k,
    )
    
    # Save results
    results_df.to_csv(args.output, index=False)
    print(f"\n✓ Saved {len(results_df)} reranked results to {args.output}")
    
    print("\nTop 10 reranked candidates:")
    print(results_df[['candidate_id', 'score', 'rerank_score', 'rerank_position']].head(10).to_string(index=False))


if __name__ == '__main__':
    main()
