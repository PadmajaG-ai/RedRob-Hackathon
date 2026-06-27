#!/usr/bin/env python3
"""
Enhanced Ranker: Hybrid retrieval using precomputed embeddings + parsing rules.
Uses GPU precomputed embeddings for semantic similarity (inference on CPU).
CPU-only suitable for final submission.
"""

import json
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import argparse

warnings.filterwarnings('ignore')

class EnhancedRanker:
    def __init__(self, precompute_dir='precompute', model_name='all-MiniLM-L6-v2'):
        """Load precomputed embeddings and initialize model."""
        self.precompute_dir = Path(precompute_dir)
        self.model_name = model_name
        
        # Load precomputed embeddings (GPU-generated, CPU inference)
        print("Loading precomputed embeddings...")
        self.candidate_embeddings = np.load(
            self.precompute_dir / 'candidate_embeddings.npy'
        )
        self.candidate_ids = np.load(
            self.precompute_dir / 'candidate_ids.npy', 
            allow_pickle=True
        )
        with open(self.precompute_dir / 'candidate_meta.json') as f:
            self.meta = json.load(f)
        
        print(f"Loaded {len(self.candidate_ids)} embeddings, dim={self.meta['dim']}")
        
        # Load model for query encoding (CPU)
        print(f"Loading model {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.model.eval()  # Inference mode
        
        # Load dataset
        self._load_dataset()
    
    def _load_dataset(self):
        """Load candidate data."""
        print("Loading dataset...")
        from eda import load_data
        self.candidates_df = load_data()
        assert len(self.candidates_df) == len(self.candidate_ids), \
            f"Mismatch: {len(self.candidates_df)} vs {len(self.candidate_ids)}"
    
    def _parse_jd(self, jd_text):
        """Extract key requirements from JD."""
        jd_lower = jd_text.lower()
        
        # Skills mentioned in JD (keywords)
        skill_keywords = {
            'python': ['python'],
            'ml': ['machine learning', 'ml', 'deep learning', 'nlp', 'cv', 'computer vision'],
            'pytorch': ['pytorch', 'torch'],
            'tensorflow': ['tensorflow', 'tf', 'keras'],
            'transformers': ['transformers', 'bert', 'gpt', 'llm'],
            'aws': ['aws', 'amazon', 's3', 'sagemaker', 'ec2'],
            'gcp': ['gcp', 'google cloud', 'bigquery'],
            'distributed': ['distributed', 'spark', 'hadoop', 'flink'],
            'data': ['data', 'sql', 'databases', 'postgresql', 'mysql'],
        }
        
        required_skills = set()
        for skill, keywords in skill_keywords.items():
            if any(kw in jd_lower for kw in keywords):
                required_skills.add(skill)
        
        # Experience requirement
        years_match = 0
        import re
        years_pattern = r'(\d+)\s*\+?\s*years'
        matches = re.findall(years_pattern, jd_lower)
        if matches:
            years_match = max(int(m) for m in matches)
        
        return {
            'required_skills': required_skills,
            'min_years': years_match,
            'jd_text': jd_text,
            'jd_text_lower': jd_lower
        }
    
    def _semantic_score(self, query_embedding, topk=100):
        """Compute semantic similarity using precomputed embeddings (CPU)."""
        # Cosine similarity: embeddings are normalized, so dot product = cosine
        similarities = cosine_similarity([query_embedding], self.candidate_embeddings)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-topk:][::-1]
        
        return {
            'indices': top_indices,
            'scores': similarities[top_indices]
        }
    
    def _extract_skills(self, skill_field):
        """Safely extract skills from candidate record."""
        if skill_field is None or (isinstance(skill_field, float) and pd.isna(skill_field)):
            return set()
        if isinstance(skill_field, dict):
            return set(skill_field.keys()) if skill_field else set()
        if isinstance(skill_field, str):
            return set(s.strip() for s in skill_field.split(',') if s.strip())
        if isinstance(skill_field, (list, tuple)):
            return set(str(s).strip().lower() for s in skill_field if s)
        return set()
    
    def _parse_experience(self, exp_field):
        """Extract years of experience."""
        if pd.isna(exp_field):
            return 0
        if isinstance(exp_field, (int, float)):
            return float(exp_field)
        
        import re
        if isinstance(exp_field, str):
            match = re.search(r'(\d+\.?\d*)', exp_field)
            if match:
                return float(match.group(1))
        return 0
    
    def rank(self, jd_text, topk=100):
        """
        Rank candidates for a JD using hybrid approach:
        1. Semantic similarity (precomputed embeddings)
        2. Skill matching
        3. Experience matching
        """
        print(f"\n{'='*60}")
        print(f"JD Ranking (top-{topk})")
        print(f"{'='*60}")
        
        # Parse JD
        jd_info = self._parse_jd(jd_text)
        required_skills = jd_info['required_skills']
        min_years = jd_info['min_years']
        
        print(f"Required skills detected: {required_skills}")
        print(f"Min years: {min_years}")
        
        # Encode JD text for semantic similarity
        print("Encoding JD text...")
        jd_embedding = self.model.encode(jd_text, convert_to_numpy=True)
        
        # Get semantic top-k
        print("Computing semantic similarity...")
        sem_result = self._semantic_score(jd_embedding, topk=topk*2)  # Get 2x to filter
        
        # Compute scores for semantic top-k
        results = []
        for idx, sem_score in zip(sem_result['indices'], sem_result['scores']):
            cand = self.candidates_df.iloc[idx]
            cand_id = self.candidate_ids[idx]
            
            # Base semantic score
            score = float(sem_score) * 50  # Scale to 0-50
            
            # Skill match bonus
            cand_skills = self._extract_skills(cand.get('skills'))
            skill_match = len(required_skills & cand_skills) / max(len(required_skills), 1)
            score += skill_match * 30  # 0-30 bonus
            
            # Experience bonus
            years = self._parse_experience(cand.get('experience'))
            exp_bonus = min(years / max(min_years, 1), 1.0) * 20  # 0-20 bonus
            score += exp_bonus
            
            results.append({
                'candidate_id': cand_id,
                'score': score,
                'semantic_score': sem_score,
                'skill_match': skill_match,
                'years_exp': years,
                'name': cand.get('name', 'N/A'),
                'location': cand.get('location', 'N/A'),
            })
        
        # Sort and return top-k
        results_df = pd.DataFrame(results).sort_values('score', ascending=False).head(topk)
        
        print(f"\nTop {len(results_df)} candidates:")
        print(results_df[['candidate_id', 'score', 'skill_match', 'years_exp', 'name']].to_string(index=False))
        
        return results_df
    
    def save_results(self, results_df, output_file):
        """Save ranking results to CSV."""
        results_df.to_csv(output_file, index=False)
        print(f"\nSaved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Enhanced semantic ranking')
    parser.add_argument('--jd', type=str, help='JD text file path')
    parser.add_argument('--jd-text', type=str, help='Direct JD text')
    parser.add_argument('--output', type=str, default='enhanced_top100.csv')
    parser.add_argument('--topk', type=int, default=100)
    parser.add_argument('--precompute-dir', type=str, default='precompute')
    args = parser.parse_args()
    
    # Load JD
    if args.jd:
        with open(args.jd) as f:
            jd_text = f.read()
    elif args.jd_text:
        jd_text = args.jd_text
    else:
        # Default JD for testing
        jd_text = """
        Senior AI Engineer - ML/NLP Focus
        
        We are looking for a Senior AI Engineer with 5+ years of experience in:
        - Machine Learning and Deep Learning
        - Natural Language Processing (NLP)
        - PyTorch or TensorFlow
        - Transformer models (BERT, GPT, etc.)
        - Python programming
        - Cloud platforms (AWS/GCP)
        
        Responsibilities:
        - Design and implement ML pipelines
        - Fine-tune large language models
        - Optimize model inference
        - Collaborate with data teams
        
        Nice to have:
        - Distributed training experience
        - LLM fine-tuning (LoRA, QLoRA)
        - MLOps and model deployment
        """
    
    # Run ranking
    ranker = EnhancedRanker(precompute_dir=args.precompute_dir)
    results = ranker.rank(jd_text, topk=args.topk)
    ranker.save_results(results, args.output)


if __name__ == '__main__':
    main()
