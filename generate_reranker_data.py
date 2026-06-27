#!/usr/bin/env python3
"""
Generate LoRA reranker training pairs from behavioral labels.

Why the previous data failed:
  synthetic_improved_training_data.json used heuristic scores as labels.
  The reranker learned to predict the heuristic → it output 0.9998 for everyone
  because the heuristic gave everyone moderate-to-high scores (no discrimination).

This fix:
  Labels come from end-of-funnel platform behavioral data:
    label = 1  if  interview_completion_rate × 0.6 + offer_acceptance_rate × 0.4  > 0.65
    label = 0  if  interview_completion_rate × 0.6 + offer_acceptance_rate × 0.4  < 0.30

  The model learns: "given this JD, which type of candidate actually completed
  interviews and accepted offers?" — a real signal, not a rule.

  Critically: the threshold gap (0.30–0.65) is intentional. It excludes ambiguous
  middle-ground candidates from training, giving the model cleaner signal on what
  clearly separates good from bad matches.

Usage:
  python generate_reranker_data.py
  python finetune_reranker.py --data reranker_training_data.json --epochs 3 \
      --batch-size 8 --output reranker_behavioral_adapter
"""

import json
import random
import argparse
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, '.')

POSITIVE_THRESHOLD = 0.65
NEGATIVE_THRESHOLD = 0.30


def behavioral_label(candidate: dict) -> float:
    def _safe(val):
        try:
            v = float(val or 0)
        except (TypeError, ValueError):
            v = 0.0
        return max(0.0, min(1.0, v))
    return _safe(candidate.get('interview_completion_rate')) * 0.6 + \
           _safe(candidate.get('offer_acceptance_rate'))     * 0.4


def candidate_to_text(candidate: dict) -> str:
    """Structured text representation fed to the cross-encoder."""
    skills = candidate.get('skills') or []
    if isinstance(skills, str):
        try:
            skills = json.loads(skills.replace("'", '"'))
        except Exception:
            skills = [s.strip() for s in skills.split(',')]
    skills_str = ', '.join(str(s) for s in skills[:20])

    return (
        f"Title: {candidate.get('current_title', '')}\n"
        f"Company: {candidate.get('current_company', '')} "
        f"({candidate.get('current_industry', '')})\n"
        f"Experience: {candidate.get('years_of_experience', 0)} years\n"
        f"Location: {candidate.get('location', '')}\n"
        f"Headline: {str(candidate.get('headline', ''))[:150]}\n"
        f"Summary: {str(candidate.get('summary', ''))[:400]}\n"
        f"Skills: {skills_str}"
    )


def generate_pairs(
    candidates,            # list of dicts or DataFrame
    jd_text: str,
    n_positive: int = 500,
    n_negative: int = 500,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> list:
    random.seed(seed)
    np.random.seed(seed)

    # Accept DataFrame or list
    if hasattr(candidates, 'to_dict'):
        records = candidates.to_dict('records')
    else:
        records = list(candidates)

    print(f"Scoring {len(records):,} candidates with behavioral label...")
    labeled = [(cand, behavioral_label(cand)) for cand in records]

    positives = [(c, l) for c, l in labeled if l > POSITIVE_THRESHOLD]
    negatives = [(c, l) for c, l in labeled if l < NEGATIVE_THRESHOLD]

    print(f"  Positives (label > {POSITIVE_THRESHOLD}): {len(positives):,}")
    print(f"  Negatives (label < {NEGATIVE_THRESHOLD}): {len(negatives):,}")
    print(f"  Excluded middle ({NEGATIVE_THRESHOLD}–{POSITIVE_THRESHOLD}): "
          f"{len(labeled) - len(positives) - len(negatives):,}  ← intentional: cleaner signal")

    n_pos = min(n_positive, len(positives))
    n_neg = min(n_negative, len(negatives))

    sampled_pos = random.sample(positives, n_pos)
    sampled_neg = random.sample(negatives, n_neg)

    # Truncate JD to leave room for candidate text within 512-token limit
    jd_short = jd_text[:900]

    pairs = []
    for cand, lbl in sampled_pos:
        pairs.append({
            'jd':               jd_short,
            'candidate':        candidate_to_text(cand),
            'label':            1,
            'behavioral_label': round(lbl, 4),
        })
    for cand, lbl in sampled_neg:
        pairs.append({
            'jd':               jd_short,
            'candidate':        candidate_to_text(cand),
            'label':            0,
            'behavioral_label': round(lbl, 4),
        })

    random.shuffle(pairs)

    val_n = max(10, int(len(pairs) * val_fraction))
    for i, pair in enumerate(pairs):
        pair['split'] = 'val' if i < val_n else 'train'

    train_n = sum(1 for p in pairs if p['split'] == 'train')
    val_n   = sum(1 for p in pairs if p['split'] == 'val')
    print(f"\nGenerated {len(pairs)} pairs  (train={train_n}, val={val_n})")
    print(f"  Positive: {n_pos}   Negative: {n_neg}")

    return pairs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jd',         default='job_description.txt')
    parser.add_argument('--output',     default='reranker_training_data.json')
    parser.add_argument('--n-positive', type=int, default=500)
    parser.add_argument('--n-negative', type=int, default=500)
    args = parser.parse_args()

    jd_text = Path(args.jd).read_text()

    from eda import load_data
    candidates = load_data()
    print(f"Loaded {len(candidates):,} candidates.\n")

    pairs = generate_pairs(candidates, jd_text, args.n_positive, args.n_negative)

    with open(args.output, 'w') as f:
        json.dump(pairs, f, indent=2)

    print(f"\n✓ Saved {len(pairs)} pairs → {args.output}")
    print(f"\nNext step — fine-tune the reranker:")
    print(f"  python finetune_reranker.py \\")
    print(f"    --data {args.output} \\")
    print(f"    --epochs 3 --batch-size 8 \\")
    print(f"    --output reranker_behavioral_adapter")
