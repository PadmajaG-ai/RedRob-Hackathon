#!/usr/bin/env python3
"""
Generate synthetic training data for reranker fine-tuning.
Creates (JD, good_candidate, bad_candidate) triplets for contrastive learning.
"""

import json
import random
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

# Try Claude API, fallback to synthetic generation
try:
    import anthropic
    HAS_CLAUDE = True
except ImportError:
    HAS_CLAUDE = False
    print("Warning: Claude API not available, using local synthetic generation")


def load_candidates():
    """Load candidate data."""
    from eda import load_data
    return load_data()


def extract_candidate_text(row):
    """Convert candidate row to readable text."""
    parts = []
    
    if row.get('current_title'):
        parts.append(f"Title: {row['current_title']}")
    if row.get('current_company'):
        parts.append(f"Company: {row['current_company']}")
    if row.get('current_industry'):
        parts.append(f"Industry: {row['current_industry']}")
    if row.get('years_of_experience'):
        parts.append(f"Experience: {row['years_of_experience']} years")
    
    skills = row.get('skills', [])
    if skills:
        if isinstance(skills, list):
            skill_str = ', '.join(str(s).strip() for s in skills[:10])
        else:
            skill_str = str(skills)[:200]
        parts.append(f"Skills: {skill_str}")
    
    if row.get('headline'):
        parts.append(f"Headline: {row['headline']}")
    if row.get('location'):
        parts.append(f"Location: {row['location']}")
    
    return " | ".join(parts)


def create_local_triplets(candidates_df, num_triplets=1000):
    """
    Create triplets locally using heuristics:
    - Good match: similar skills + relevant experience
    - Bad match: unrelated skills or insufficient experience
    """
    print(f"Creating {num_triplets} synthetic triplets locally...")
    
    # Common JD profiles for recruiting
    jd_templates = [
        {
            'title': 'Senior AI/ML Engineer',
            'skills': ['machine learning', 'python', 'pytorch', 'transformers', 'deep learning'],
            'years': 5,
            'industries': ['tech', 'AI', 'fintech'],
        },
        {
            'title': 'Data Scientist',
            'skills': ['data science', 'python', 'sql', 'statistics', 'ml'],
            'years': 4,
            'industries': ['tech', 'fintech', 'analytics'],
        },
        {
            'title': 'Backend Engineer',
            'skills': ['python', 'java', 'go', 'system design', 'databases'],
            'years': 5,
            'industries': ['tech', 'fintech', 'e-commerce'],
        },
        {
            'title': 'DevOps Engineer',
            'skills': ['kubernetes', 'docker', 'aws', 'ci/cd', 'python'],
            'years': 4,
            'industries': ['tech', 'fintech', 'cloud'],
        },
        {
            'title': 'Full Stack Engineer',
            'skills': ['javascript', 'react', 'node.js', 'databases', 'aws'],
            'years': 3,
            'industries': ['tech', 'startup', 'e-commerce'],
        },
    ]
    
    triplets = []
    candidate_list = candidates_df.to_dict('records')
    
    for _ in range(num_triplets):
        # Pick random JD template
        jd_template = random.choice(jd_templates)
        
        jd_text = f"Role: {jd_template['title']} | Required skills: {', '.join(jd_template['skills'])} | Experience: {jd_template['years']}+ years"
        
        # Find good match: has relevant skills + experience
        good_cand = None
        for _ in range(10):  # Try 10 times
            cand = random.choice(candidate_list)
            cand_skills = cand.get('skills', [])
            if not isinstance(cand_skills, list):
                cand_skills = []
            
            cand_exp = float(cand.get('years_of_experience') or 0)
            
            # Check skill overlap
            skill_overlap = len(set(str(s).lower() for s in cand_skills) & 
                               set(s.lower() for s in jd_template['skills']))
            
            if skill_overlap >= 2 and cand_exp >= jd_template['years'] - 2:
                good_cand = cand
                break
        
        # Find bad match: different skills or insufficient experience
        bad_cand = None
        for _ in range(10):
            cand = random.choice(candidate_list)
            cand_skills = cand.get('skills', [])
            if not isinstance(cand_skills, list):
                cand_skills = []
            
            cand_exp = float(cand.get('years_of_experience') or 0)
            
            # Check skill overlap
            skill_overlap = len(set(str(s).lower() for s in cand_skills) & 
                               set(s.lower() for s in jd_template['skills']))
            
            if skill_overlap <= 1 or cand_exp < jd_template['years'] - 4:
                bad_cand = cand
                break
        
        if good_cand and bad_cand:
            triplets.append({
                'jd': jd_text,
                'good_candidate': extract_candidate_text(good_cand),
                'bad_candidate': extract_candidate_text(bad_cand),
                'label': 1,  # For labeled pairs
            })
    
    return triplets


def create_claude_triplets(num_triplets=500):
    """
    Generate synthetic triplets using Claude API.
    Each call creates (JD, good_profile, bad_profile) triplet.
    """
    if not HAS_CLAUDE:
        print("Claude API not available")
        return []
    
    client = anthropic.Anthropic()
    triplets = []
    
    print(f"Generating {num_triplets} triplets via Claude API...")
    
    for i in tqdm(range(num_triplets)):
        try:
            message = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Generate a recruiting triplet (JSON) with this format:
{{
  "jd": "Job Description for a Senior AI Engineer role with 5+ years ML/Python experience",
  "good_candidate": "Profile of someone who matches the JD well",
  "bad_candidate": "Profile of someone who doesn't match well"
}}

Make profiles realistic (name, title, company, skills, experience years, location).
Ensure good_candidate has relevant skills and experience.
Ensure bad_candidate is clearly mismatched (wrong skills or too junior).

Return only valid JSON, no markdown."""
                    }
                ]
            )
            
            content = message.content[0].text
            # Try to parse JSON
            try:
                data = json.loads(content)
                data['label'] = 1
                triplets.append(data)
            except json.JSONDecodeError:
                # Try to extract JSON from content
                start = content.find('{')
                end = content.rfind('}') + 1
                if start >= 0 and end > start:
                    try:
                        data = json.loads(content[start:end])
                        data['label'] = 1
                        triplets.append(data)
                    except:
                        pass
        
        except anthropic.RateLimitError:
            print(f"Rate limit hit at triplet {i}. Stopping generation.")
            break
        except Exception as e:
            print(f"Error generating triplet {i}: {e}")
            continue
    
    print(f"Generated {len(triplets)} triplets via Claude")
    return triplets


def convert_to_pairs(triplets):
    """Convert triplets to positive/negative pairs for contrastive learning."""
    pairs = []
    
    for triplet in triplets:
        # Positive pair: (JD, good_candidate)
        pairs.append({
            'jd': triplet['jd'],
            'candidate': triplet['good_candidate'],
            'label': 1,
        })
        
        # Negative pair: (JD, bad_candidate)
        pairs.append({
            'jd': triplet['jd'],
            'candidate': triplet['bad_candidate'],
            'label': 0,
        })
    
    return pairs


def save_dataset(pairs, output_file='synthetic_training_data.json'):
    """Save dataset to JSON."""
    with open(output_file, 'w') as f:
        json.dump({
            'metadata': {
                'total_pairs': len(pairs),
                'positive': sum(1 for p in pairs if p['label'] == 1),
                'negative': sum(1 for p in pairs if p['label'] == 0),
            },
            'pairs': pairs
        }, f, indent=2)
    
    print(f"Saved {len(pairs)} pairs to {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic reranker training data')
    parser.add_argument('--local-triplets', type=int, default=500,
                       help='Number of triplets to generate locally')
    parser.add_argument('--claude-triplets', type=int, default=0,
                       help='Number of triplets to generate via Claude API')
    parser.add_argument('--output', type=str, default='synthetic_training_data.json',
                       help='Output file')
    args = parser.parse_args()
    
    triplets = []
    
    # Generate local triplets
    if args.local_triplets > 0:
        candidates_df = load_candidates()
        local_triplets = create_local_triplets(candidates_df, args.local_triplets)
        triplets.extend(local_triplets)
    
    # Generate Claude triplets
    if args.claude_triplets > 0:
        claude_triplets = create_claude_triplets(args.claude_triplets)
        triplets.extend(claude_triplets)
    
    if not triplets:
        print("No triplets generated!")
        return
    
    # Convert to pairs
    pairs = convert_to_pairs(triplets)
    
    # Save
    output_file = save_dataset(pairs, args.output)
    
    print(f"\nGenerated {len(pairs)} pairs ({len(triplets)} triplets)")
    print(f"  Positive pairs: {sum(1 for p in pairs if p['label'] == 1)}")
    print(f"  Negative pairs: {sum(1 for p in pairs if p['label'] == 0)}")
    print(f"  Saved to: {output_file}")


if __name__ == '__main__':
    main()
