#!/usr/bin/env python3
"""
Full DoRA/LoRA fine-tuning workflow for recruiting ranking.
Steps:
  1. Generate synthetic training data
  2. Fine-tune reranker with DoRA (or LoRA fallback)
  3. Test on CPU inference
  4. Create final ranked results
"""

import os
import subprocess
import argparse
from pathlib import Path
import json

def run_command(cmd, description=""):
    """Run shell command with error handling."""
    print(f"\n{'='*70}")
    if description:
        print(f"📍 {description}")
    print(f"{'='*70}")
    print(f"$ {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Command failed with exit code {result.returncode}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description='Full DoRA fine-tuning workflow')
    parser.add_argument('--step', type=int, default=0, 
                       help='Start from step (0=all, 1=data, 2=finetune, 3=rerank)')
    parser.add_argument('--local-triplets', type=int, default=1000,
                       help='Local triplets for training data')
    parser.add_argument('--claude-triplets', type=int, default=0,
                       help='Claude API triplets (requires API key)')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Fine-tuning epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--rank', type=int, default=8,
                       help='LoRA rank')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show commands without running')
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                 DoRA/LoRA Fine-tuning Workflow                   ║
║              For Recruiting Candidate Ranking                    ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    steps = {
        1: ("Generate Synthetic Training Data", f"""
python generate_synthetic_data.py \\
  --local-triplets {args.local_triplets} \\
  --claude-triplets {args.claude_triplets} \\
  --output synthetic_training_data.json
        """),
        
        2: ("Fine-tune Reranker with DoRA/LoRA", f"""
python finetune_reranker.py \\
  --data synthetic_training_data.json \\
  --model BAAI/bge-reranker-v2-m3 \\
  --use-dora \\
  --rank {args.rank} \\
  --epochs {args.epochs} \\
  --batch-size {args.batch_size} \\
  --output reranker_adapter \\
  --lr 2e-4
        """),
        
        3: ("CPU Reranking with Adapter", """
python rerank_cpu.py \\
  --model BAAI/bge-reranker-v2-m3 \\
  --adapter reranker_adapter \\
  --input-csv enhanced_top100.csv \\
  --output reranked_results.csv \\
  --top-k 100
        """),
    }
    
    print("\n📋 WORKFLOW STEPS:")
    print("-" * 70)
    for step_num, (desc, cmd) in steps.items():
        status = "→" if step_num >= args.step else "✓"
        print(f"{status} Step {step_num}: {desc}")
    print("-" * 70)
    
    print("\n📋 CONFIGURATION:")
    print(f"  Local triplets: {args.local_triplets}")
    print(f"  Claude triplets: {args.claude_triplets}")
    print(f"  Fine-tune epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LoRA rank: {args.rank}")
    print(f"  Dry run: {args.dry_run}")
    
    if args.dry_run:
        print("\n🔍 DRY RUN - Commands to execute:")
        print("=" * 70)
        for step_num in range(args.step, 4):
            if step_num in steps:
                _, cmd = steps[step_num]
                print(f"\nStep {step_num}:")
                print(cmd)
        return
    
    # Execute steps
    success = True
    for step_num in range(args.step, 4):
        if step_num not in steps:
            continue
        
        desc, cmd = steps[step_num]
        
        # Check prerequisites
        if step_num == 2:
            if not Path('synthetic_training_data.json').exists():
                print(f"⚠️  Step 1 (data generation) must complete first")
                continue
        elif step_num == 3:
            if not Path('reranker_adapter').exists():
                print(f"⚠️  Step 2 (fine-tuning) must complete first")
                continue
        
        success = run_command(cmd.strip(), desc)
        if not success:
            print(f"\n❌ Workflow interrupted at Step {step_num}")
            break
        
        print(f"✓ Step {step_num} complete!")
    
    if success:
        print("""
╔══════════════════════════════════════════════════════════════════╗
║                  ✅ WORKFLOW COMPLETE!                           ║
╚══════════════════════════════════════════════════════════════════╝

📊 ARTIFACTS CREATED:
  1. synthetic_training_data.json   - Training pairs
  2. reranker_adapter/              - Fine-tuned DoRA/LoRA adapter
  3. reranked_results.csv           - Final CPU-inferred rankings

🚀 NEXT STEPS:
  • Review reranked_results.csv
  • Compare with enhanced_top100.csv to see improvement
  • Deploy reranked_results.csv as final submission

💡 TO USE ADAPTER ON CPU:
  from transformers import AutoTokenizer, AutoModelForSequenceClassification
  from peft import PeftModel
  
  model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3')
  model = PeftModel.from_pretrained(model, 'reranker_adapter')
  tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')
  
  # Inference on CPU - no GPU needed!
  inputs = tokenizer("JD text", "candidate text", return_tensors="pt")
  outputs = model(**inputs)
  score = outputs.logits.softmax(dim=1)[0, 1].item()
        """)
    else:
        print("\n❌ Workflow failed")


if __name__ == '__main__':
    main()
