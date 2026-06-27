import argparse
import csv
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

ARCHIVE = Path('[PUB] India_runs_data_and_ai_challenge.zip')
INTERNAL_JSON = 'India_runs_data_and_ai_challenge/candidates.jsonl'
INTERNAL_JD = 'India_runs_data_and_ai_challenge/job_description.docx'

TARGET_CITY_KEYWORDS = [
    'pune', 'noida', 'delhi', 'mumbai', 'bangalore', 'hyderabad', 'chennai',
    'kolkata', 'ahmedabad', 'jaipur', 'gurgaon', 'gurugram', 'bengaluru', 'trivandr',
    'kochi', 'mumbai', 'pune', 'chandigarh', 'dehradun', 'lucknow', 'vadodara',
]

POSITIVE_TITLE_KEYWORDS = [
    'engineer', 'developer', 'researcher', 'scientist', 'ml', 'ai', 'data',
    'analytics', 'architect', 'systems', 'algorithm', 'backend', 'frontend',
    'software', 'devops', 'platform', 'machine learning', 'deep learning',
]
NEGATIVE_TITLE_KEYWORDS = [
    'manager', 'assistant', 'accountant', 'sales', 'support', 'hr', 'teacher',
    'designer', 'executive', 'consultant', 'intern', 'recruiter', 'operations',
    'marketing', 'content', 'customer', 'business analyst', 'business development',
]

JD_KEYWORDS = [
    'ai', 'ml', 'machine learning', 'deep learning', 'llm', 'retrieval', 'ranking',
    'embeddings', 'vector search', 'fine tuning', 'fine-tuning', 'product',
    'startup', 'founding', 'ship', 'production', 'systems', 'search', 'workflow',
    'performance', 'scaling', 'pipelines', 'backend', 'recommendation', 'evaluation',
]


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
    return re.sub(r'\s+', ' ', text.strip().lower())


def extract_candidate_text(record: dict) -> str:
    prof = record['profile']
    skills = record.get('skills', [])
    skill_terms = []
    for item in skills:
        if isinstance(item, str):
            skill_terms.append(item)
        elif isinstance(item, dict) and item.get('name'):
            skill_terms.append(item['name'])
    skill_text = ' '.join(skill_terms)
    title = prof.get('current_title', '')
    headline = prof.get('headline', '')
    summary = prof.get('summary', '')
    industry = prof.get('current_industry', '')
    company = prof.get('current_company', '')
    return normalize_text(' '.join([title, headline, summary, skill_text, industry, company]))


def title_score(title: str) -> float:
    normalized = normalize_text(title)
    score = 0.0
    for term in POSITIVE_TITLE_KEYWORDS:
        if term in normalized:
            score += 1.0
    for term in NEGATIVE_TITLE_KEYWORDS:
        if term in normalized:
            score -= 0.75
    return max(score, 0.0)


def keyword_score(text: str) -> float:
    score = 0.0
    for term in JD_KEYWORDS:
        if term in text:
            score += 1.0
    return min(score / max(len(JD_KEYWORDS), 1), 1.0)


def location_score(record: dict) -> float:
    profile = record['profile']
    signals = record['redrob_signals']
    location = normalize_text(profile.get('location', ''))
    score = 0.0
    for city in TARGET_CITY_KEYWORDS:
        if city in location:
            score = 1.0
            break
    if score == 0.0 and signals.get('willing_to_relocate'):
        score = 0.8
    return score


def normalize_signal(values, missing_value=0.0):
    arr = np.array([v if v is not None else np.nan for v in values], dtype=float)
    if np.all(np.isnan(arr)):
        return np.full_like(arr, missing_value)
    min_val = np.nanmin(arr)
    max_val = np.nanmax(arr)
    if min_val == max_val:
        return np.nan_to_num(arr, nan=missing_value)
    scaled = (arr - min_val) / (max_val - min_val)
    return np.nan_to_num(scaled, nan=missing_value)


def build_behavior_score(candidates: list[dict]) -> np.ndarray:
    recruiter_rate = [c['redrob_signals'].get('recruiter_response_rate') for c in candidates]
    response_time = [c['redrob_signals'].get('avg_response_time_hours') for c in candidates]
    profile_complete = [c['redrob_signals'].get('profile_completeness_score') for c in candidates]
    search_appear = [c['redrob_signals'].get('search_appearance_30d') for c in candidates]
    saved_by = [c['redrob_signals'].get('saved_by_recruiters_30d') for c in candidates]
    interview_complete = [c['redrob_signals'].get('interview_completion_rate') for c in candidates]
    offer_accept = [c['redrob_signals'].get('offer_acceptance_rate') for c in candidates]
    github_score = [c['redrob_signals'].get('github_activity_score') for c in candidates]

    rr_norm = normalize_signal(recruiter_rate)
    rt_norm = 1.0 - normalize_signal(response_time)
    pc_norm = normalize_signal(profile_complete)
    sa_norm = normalize_signal(search_appear)
    sb_norm = normalize_signal(saved_by)
    ic_norm = normalize_signal(interview_complete)
    oa_norm = normalize_signal([((v + 1.0) / 2.0) if v is not None else None for v in offer_accept])
    gh_norm = normalize_signal([0.0 if v == -1 else v for v in github_score])

    behavior = (
        0.25 * rr_norm
        + 0.15 * rt_norm
        + 0.15 * pc_norm
        + 0.15 * sa_norm
        + 0.10 * sb_norm
        + 0.10 * ic_norm
        + 0.05 * oa_norm
        + 0.05 * gh_norm
    )
    return np.clip(behavior, 0.0, 1.0)


def experience_score(record: dict) -> float:
    years = record['profile'].get('years_of_experience')
    if years is None:
        return 0.2
    if 5.0 <= years <= 9.0:
        return 1.0
    if 3.0 <= years < 5.0 or 9.0 < years <= 11.0:
        return 0.6
    return 0.2


def notice_penalty(record: dict) -> float:
    notice = record['redrob_signals'].get('notice_period_days')
    if notice is None:
        return 1.0
    if notice <= 30:
        return 1.0
    if notice <= 90:
        return 0.8
    return 0.6


def salary_penalty(record: dict) -> float:
    salary_info = record['redrob_signals'].get('expected_salary_range_inr_lpa') or {}
    max_sal = salary_info.get('max')
    if max_sal is None:
        return 1.0
    if max_sal <= 25:
        return 1.0
    if max_sal <= 35:
        return 0.9
    return 0.8


def compute_candidate_scores(candidate_texts, query_text, records):
    vectorizer = TfidfVectorizer(max_features=12000, stop_words='english', ngram_range=(1, 2))
    candidates_matrix = vectorizer.fit_transform(candidate_texts)
    query_vec = vectorizer.transform([query_text])
    similarity = (candidates_matrix @ query_vec.T).toarray().flatten()
    similarity = normalize_signal(similarity)

    behavior_scores = build_behavior_score(records)

    scores = []
    for idx, record in enumerate(records):
        text = candidate_texts[idx]
        sim = float(similarity[idx])
        kw = keyword_score(text)
        title = title_score(record['profile'].get('current_title', ''))
        exp = experience_score(record)
        loc = location_score(record)
        notice = notice_penalty(record)
        salary = salary_penalty(record)
        behavior = float(behavior_scores[idx])

        score = (
            0.35 * sim
            + 0.20 * kw
            + 0.15 * title
            + 0.10 * exp
            + 0.10 * behavior
            + 0.05 * loc
            + 0.025 * notice
            + 0.025 * salary
        )
        scores.append(score)
    return np.array(scores, dtype=float)


def read_candidates() -> list[dict]:
    with zipfile.ZipFile(ARCHIVE) as zf:
        with zf.open(INTERNAL_JSON) as f:
            return [json.loads(line) for line in f]


def build_top_candidates(records, scores, n=100):
    order = np.argsort(-scores)
    top_idx = order[:n]
    top_rows = []
    current = 1
    previous_score = None
    for rank, idx in enumerate(top_idx, start=1):
        score = float(scores[idx])
        if previous_score is not None and score > previous_score:
            score = previous_score
        previous_score = score
        record = records[idx]
        top_rows.append(
            {
                'candidate_id': record['candidate_id'],
                'rank': rank,
                'score': round(score, 6),
                'reasoning': generate_reasoning(record, score),
            }
        )
    return top_rows


def generate_reasoning(record: dict, score: float) -> str:
    prof = record['profile']
    signals = record['redrob_signals']
    pieces = []
    pieces.append(f"{prof.get('current_title','Unknown')} with {prof.get('years_of_experience','?')} yrs exp")
    if signals.get('open_to_work_flag'):
        pieces.append('open to work')
    if signals.get('recruiter_response_rate', 0) >= 0.5:
        pieces.append('good recruiter response')
    if signals.get('search_appearance_30d', 0) >= 50:
        pieces.append('recent search visibility')
    return '; '.join(pieces)[:240]


def write_submission(rows, output_path: Path):
    with output_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['candidate_id', 'rank', 'score', 'reasoning'])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description='Hybrid ranker for the Redrob challenge')
    parser.add_argument('--output', type=Path, default=Path('hybrid_top100.csv'))
    parser.add_argument('--topk', type=int, default=100)
    args = parser.parse_args()

    print('Reading job description...')
    job_text = normalize_text(read_docx_text(ARCHIVE, INTERNAL_JD))
    print('Reading candidates...')
    records = read_candidates()
    print('Preparing candidate texts...')
    candidate_texts = [extract_candidate_text(r) for r in records]
    print('Building candidate scores... this may take a couple minutes')
    scores = compute_candidate_scores(candidate_texts, job_text, records)
    print('Selecting top candidates...')
    top_rows = build_top_candidates(records, scores, args.topk)
    write_submission(top_rows, args.output)
    print('Wrote', args.output)


if __name__ == '__main__':
    main()
