import argparse
import json
import math
import os
import zipfile
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

ARCHIVE = Path('[PUB] India_runs_data_and_ai_challenge.zip')
INTERNAL_JSON = 'India_runs_data_and_ai_challenge/candidates.jsonl'
INTERNAL_JD = 'India_runs_data_and_ai_challenge/job_description.docx'


def read_docx_text(zip_path: Path, inner_path: str) -> str:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(inner_path) as docx:
            import io
            import xml.etree.ElementTree as ET
            with zipfile.ZipFile(io.BytesIO(docx.read())) as z2:
                xml = z2.read('word/document.xml')
            root = ET.fromstring(xml)
            text = []
            for elem in root.iter():
                if elem.tag.endswith('}t'):
                    text.append(elem.text or '')
                elif elem.tag.endswith('}tab'):
                    text.append('\t')
                elif elem.tag.endswith('}br') or elem.tag.endswith('}cr'):
                    text.append('\n')
            return ''.join(text)


def normalize_text(text: str) -> str:
    return ' '.join(text.strip().lower().split())


def extract_candidate_text(record: dict) -> str:
    profile = record['profile']
    title = profile.get('current_title', '')
    headline = profile.get('headline', '')
    summary = profile.get('summary', '')
    industry = profile.get('current_industry', '')
    company = profile.get('current_company', '')
    skills = []
    for item in record.get('skills', []):
        if isinstance(item, str):
            skills.append(item)
        elif isinstance(item, dict) and item.get('name'):
            skills.append(item['name'])
    candidate_text = ' '.join([title, headline, summary, industry, company, ' '.join(skills)])
    return normalize_text(candidate_text)


def load_candidates() -> list[dict]:
    with zipfile.ZipFile(ARCHIVE) as zf:
        with zf.open(INTERNAL_JSON) as f:
            return [json.loads(line) for line in f]


def build_embeddings(model_name: str, texts: list[str], batch_size: int, device: str) -> np.ndarray:
    model = SentenceTransformer(model_name, device=device)
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc='Embedding', unit='batch'):
        batch = texts[i:i + batch_size]
        emb = model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
        embeddings.append(emb.cpu().numpy())
    return np.vstack(embeddings)


def main():
    parser = argparse.ArgumentParser(description='Precompute candidate embeddings for Redrob dataset')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2', help='SentenceTransformer model name')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--output-dir', type=Path, default=Path('precompute'))
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Using device: {args.device}')

    print('Loading candidates...')
    candidates = load_candidates()
    texts = [extract_candidate_text(rec) for rec in candidates]
    ids = [rec['candidate_id'] for rec in candidates]

    print('Building embeddings...')
    embeddings = build_embeddings(args.model, texts, args.batch_size, args.device)
    np.save(args.output_dir / 'candidate_ids.npy', np.array(ids, dtype=object))
    np.save(args.output_dir / 'candidate_embeddings.npy', embeddings)
    with open(args.output_dir / 'candidate_meta.json', 'w', encoding='utf-8') as f:
        json.dump({'count': len(ids), 'dim': embeddings.shape[1], 'model': args.model}, f)
    print('Saved embeddings and metadata to', args.output_dir)


if __name__ == '__main__':
    main()
