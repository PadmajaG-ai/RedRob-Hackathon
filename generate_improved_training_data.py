#!/usr/bin/env python3
"""
Improved Training Data Generation
Uses behavioral signals + skill matching for higher-quality training pairs
"""
import json
import random
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from eda import load_data


def score_candidate_for_jd(candidate: dict, jd_requirements: dict) -> float:
    """
    Score how well a candidate matches JD requirements.
    Uses multiple signals: skills, experience, behavioral signals, industry match.
    
    Returns score 0.0-1.0
    """
    score = 0.0
    
    # 1. Skill match (weight: 0.40)
    required_skills = set(jd_requirements.get('skills', []))
    candidate_skills = set([s.lower() for s in candidate.get('skills', [])])
    if required_skills:
        skill_overlap = len(required_skills & candidate_skills) / len(required_skills)
        score += 0.40 * skill_overlap
    
    # 2. Years of experience (weight: 0.20)
    required_years = jd_requirements.get('years', 0)
    candidate_years = candidate.get('years_of_experience') or 0
    years_match = min(candidate_years / max(required_years, 1), 1.0)
    score += 0.20 * years_match
    
    # 3. Industry match (weight: 0.15)
    preferred_industries = jd_requirements.get('industries', [])
    candidate_industry = candidate.get('current_industry', '').lower()
    industry_match = 1.0 if any(ind.lower() in candidate_industry for ind in preferred_industries) else 0.5
    score += 0.15 * industry_match
    
    # 4. Behavioral signals (weight: 0.25)
    # Normalize signals to 0-1 range
    response_rate = min((candidate.get('recruiter_response_rate') or 0) / 0.9, 1.0)
    open_to_work = 1.0 if candidate.get('open_to_work_flag') else 0.5
    interview_rate = min((candidate.get('interview_completion_rate') or 0) / 0.7, 1.0)
    
    behavioral_score = np.mean([response_rate, open_to_work, interview_rate])
    score += 0.25 * behavioral_score
    
    return min(score, 1.0)


def create_improved_triplets(candidates_df: pd.DataFrame, num_triplets: int = 500) -> list:
    """
    Create high-quality triplets using multi-signal matching.
    
    For each triplet:
    - Anchor: JD requirement
    - Positive: candidate with high match score
    - Negative: candidate with low match score
    """
    
    jds = [
        {
            'title': 'ML Engineer',
            'skills': ['machine learning', 'python', 'tensorflow', 'pytorch', 'nlp'],
            'years': 4,
            'industries': ['tech', 'ai', 'fintech'],
            'description': 'Senior ML Engineer for LLM applications'
        },
        {
            'title': 'Data Engineer',
            'skills': ['python', 'sql', 'spark', 'airflow', 'databases'],
            'years': 5,
            'industries': ['tech', 'fintech', 'data'],
            'description': 'Data Engineer to build ETL pipelines'
        },
        {
            'title': 'Backend Engineer',
            'skills': ['python', 'java', 'go', 'system design', 'databases', 'api'],
            'years': 5,
            'industries': ['tech', 'fintech', 'e-commerce'],
            'description': 'Backend Engineer for scalable systems'
        },
        {
            'title': 'Full Stack Engineer',
            'skills': ['javascript', 'react', 'node.js', 'databases', 'aws', 'css'],
            'years': 3,
            'industries': ['tech', 'startup', 'e-commerce'],
            'description': 'Full Stack Engineer for web applications'
        },
        {
            'title': 'DevOps Engineer',
            'skills': ['kubernetes', 'docker', 'aws', 'ci/cd', 'linux'],
            'years': 4,
            'industries': ['tech', 'cloud', 'infrastructure'],
            'description': 'DevOps Engineer for infrastructure automation'
        },
        {
            'title': 'Product Manager',
            'skills': ['product management', 'analytics', 'sql', 'communication'],
            'years': 5,
            'industries': ['tech', 'startup', 'saas'],
            'description': 'Product Manager for B2B SaaS'
        },
    ]
    
    triplets = []
    
    # Score all candidates once
    print("Scoring all candidates against JDs...")
    candidate_scores = {}  # jd_idx -> list of (cand_id, score)
    
    for jd_idx, jd in enumerate(jds):
        scores = []
        for idx, row in candidates_df.iterrows():
            candidate = row.to_dict()
            score = score_candidate_for_jd(candidate, jd)
            scores.append((candidate['candidate_id'], score, idx))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        candidate_scores[jd_idx] = scores
    
    print(f"Creating {num_triplets} triplets...")
    
    for i in range(num_triplets):
        jd_idx = i % len(jds)
        jd = jds[jd_idx]
        scores = candidate_scores[jd_idx]
        
        if len(scores) < 2:
            continue
        
        # Positive: top 30% candidates
        pos_idx = random.randint(0, max(1, len(scores) // 3))
        pos_id, pos_score, _ = scores[pos_idx]
        
        # Negative: bottom 30% candidates
        neg_idx = random.randint(max(len(scores) * 2 // 3, len(scores) - 1), len(scores) - 1)
        neg_id, neg_score, _ = scores[neg_idx]
        
        # Skip if scores are too similar
        if abs(pos_score - neg_score) < 0.15:
            continue
        
        # Get full candidate text from dataframe
        pos_cand = candidates_df[candidates_df['candidate_id'] == pos_id].iloc[0] if len(candidates_df[candidates_df['candidate_id'] == pos_id]) > 0 else None
        neg_cand = candidates_df[candidates_df['candidate_id'] == neg_id].iloc[0] if len(candidates_df[candidates_df['candidate_id'] == neg_id]) > 0 else None
        
        if pos_cand is None or neg_cand is None:
            continue
        
        # Format JD text
        jd_text = f"{jd['description']}. Required: {', '.join(jd['skills'][:3])}. Years: {jd['years']}+."
        
        # Format candidate texts
        def format_candidate(cand):
            return f"{cand.get('current_title', 'Unknown')} at {cand.get('current_company', 'Unknown')} ({cand.get('years_of_experience', 0):.1f} yrs). Skills: {', '.join(str(s)[:20] for s in (cand.get('skills', [])[:5]))}. Response rate: {cand.get('recruiter_response_rate', 0):.2f}. Open to work: {cand.get('open_to_work_flag', False)}."
        
        triplet = {
            'jd': jd_text,
            'positive': format_candidate(pos_cand),
            'negative': format_candidate(neg_cand),
            'jd_title': jd['title'],
            'positive_score': float(pos_score),
            'negative_score': float(neg_score),
            'positive_id': pos_id,
            'negative_id': neg_id,
        }
        
        triplets.append(triplet)
    
    return triplets


def triplets_to_pairs(triplets: list) -> list:
    """Convert triplets to (JD, candidate, label) pairs."""
    pairs = []
    for triplet in triplets:
        # Positive pair
        pairs.append({
            'jd': triplet['jd'],
            'candidate': triplet['positive'],
            'label': 1,
            'source': 'improved_triplet',
            'quality_score': triplet['positive_score'],
        })
        # Negative pair
        pairs.append({
            'jd': triplet['jd'],
            'candidate': triplet['negative'],
            'label': 0,
            'source': 'improved_triplet',
            'quality_score': triplet['negative_score'],
        })
    
    return pairs


def save_dataset(pairs: list, output_file: str = 'synthetic_improved_training_data.json'):
    """Save pairs to JSON."""
    with open(output_file, 'w') as f:
        json.dump(pairs, f, indent=2)
    print(f"✓ Saved {len(pairs)} pairs to {output_file}")
    return output_file


def main():
    print("Loading candidate data...")
    candidates_df = load_data()
    print(f"Loaded {len(candidates_df)} candidates")
    
    print("\nGenerating improved training triplets...")
    triplets = create_improved_triplets(candidates_df, num_triplets=500)
    print(f"✓ Generated {len(triplets)} triplets")
    
    pairs = triplets_to_pairs(triplets)
    print(f"✓ Converted to {len(pairs)} pairs")
    
    # Split into train/val
    random.shuffle(pairs)
    split_idx = int(0.9 * len(pairs))
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    
    # Add metadata
    for p in pairs:
        p['split'] = 'train' if p in train_pairs else 'val'
    
    output_file = save_dataset(pairs)
    
    print(f"\nDataset split:")
    print(f"  Training: {len(train_pairs)} pairs")
    print(f"  Validation: {len(val_pairs)} pairs")
    
    # Show sample
    print(f"\nSample pair:")
    print(f"  JD: {pairs[0]['jd'][:80]}...")
    print(f"  Candidate: {pairs[0]['candidate'][:80]}...")
    print(f"  Label: {pairs[0]['label']}")
    
    return output_file


if __name__ == '__main__':
    main()
