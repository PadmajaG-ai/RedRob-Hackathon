#!/usr/bin/env python3
"""
LLM-as-Judge: Multi-dimensional candidate scoring with per-candidate reasoning.

Two backends:
  --mode heuristic   Fast rule-based scoring, structured JSON output (no model needed)
  --mode llm         Local instruction-following model via HuggingFace transformers

Scoring dimensions (0-10 each):
  1. technical_skill_match   — depth + recency of required skills
  2. career_trajectory       — seniority, progression, growth rate
  3. domain_relevance        — industry verticals, company size match
  4. behavioral_engagement   — platform activity, responsiveness, contributions
  5. cultural_fit            — availability signals, work mode alignment
  6. bonus_signals           — GitHub activity, search presence, offer acceptance

Final score = weighted sum (weights configurable per role type).
"""

import json
import re
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

# ──────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────

@dataclass
class DimensionScores:
    technical_skill_match: float     # 0-10
    career_trajectory: float         # 0-10
    domain_relevance: float          # 0-10
    behavioral_engagement: float     # 0-10
    cultural_fit: float              # 0-10
    bonus_signals: float             # 0-10


# Default weights for IC (individual contributor) roles
IC_WEIGHTS = {
    'technical_skill_match': 0.30,
    'career_trajectory':     0.20,
    'domain_relevance':      0.20,
    'behavioral_engagement': 0.15,
    'cultural_fit':          0.10,
    'bonus_signals':         0.05,
}

LEAD_WEIGHTS = {
    'technical_skill_match': 0.20,
    'career_trajectory':     0.30,
    'domain_relevance':      0.20,
    'behavioral_engagement': 0.15,
    'cultural_fit':          0.10,
    'bonus_signals':         0.05,
}

ROLE_WEIGHTS = {'ic': IC_WEIGHTS, 'lead': LEAD_WEIGHTS}

# Synonym map for flexible skill matching.
# Key = canonical name as it might appear in a JD skill list.
# Value = list of equivalent terms that appear in candidate profiles.
SKILL_SYNONYMS = {
    'python':               ['py'],
    'pytorch':              ['torch'],
    'tensorflow':           ['tf', 'keras'],
    'scikit-learn':         ['sklearn'],
    'machine learning':     ['ml'],
    'deep learning':        ['dl', 'neural network', 'neural networks'],
    'nlp':                  ['natural language processing', 'text processing', 'computational linguistics'],
    'llm':                  ['large language model', 'llms', 'language model', 'gpt'],
    'llms':                 ['llm', 'large language model', 'language model'],
    'transformers':         ['bert', 'gpt', 'transformer', 'attention mechanism'],
    'embeddings':           ['embedding', 'vector embedding', 'dense vector', 'text embedding'],
    'retrieval':            ['information retrieval', 'ir', 'document retrieval'],
    'rag':                  ['retrieval augmented generation', 'retrieval-augmented generation'],
    'fine-tuning':          ['finetuning', 'fine tuning', 'model adaptation', 'model training'],
    'lora':                 ['low-rank adaptation', 'adapter tuning', 'lora fine-tuning'],
    'qlora':                ['quantized lora', 'quantized fine-tuning'],
    'vector search':        ['vector database', 'vector store', 'ann', 'similarity search', 'approximate nearest neighbor'],
    'elasticsearch':        ['elastic', 'elastic search'],
    'faiss':                ['vector index', 'facebook ai similarity search'],
    'learning to rank':     ['ltr', 'ranknet', 'lambdamart', 'listwise ranking'],
    'reranking':            ['reranker', 're-ranking', 'cross-encoder', 'neural reranking'],
    'hybrid search':        ['hybrid retrieval', 'dense sparse fusion', 'bm25 fusion'],
    'semantic search':      ['dense retrieval', 'neural search', 'embedding search'],
    'ndcg':                 ['normalized discounted cumulative gain', 'ranking metric'],
    'mrr':                  ['mean reciprocal rank', 'ranking evaluation'],
    'mlops':                ['ml ops', 'ml infrastructure', 'model serving', 'model deployment'],
    'rest api':             ['api', 'restful', 'web api', 'http api'],
    'microservices':        ['microservice', 'distributed systems', 'service oriented'],
}


# ──────────────────────────────────────────────
# JD parser
# ──────────────────────────────────────────────

def parse_jd(jd_text: str) -> dict:
    """Extract structured requirements from JD text."""
    jd_lower = jd_text.lower()

    # Known tech skills to detect in JD text
    # Note: 'computer vision', 'speech', 'robotics' excluded — often appear as disqualifiers
    KNOWN_SKILLS = [
        'python', 'java', 'scala', 'go', 'rust', 'c++', 'javascript', 'typescript',
        'machine learning', 'deep learning', 'nlp', 'llm', 'llms', 'transformers',
        'pytorch', 'tensorflow', 'keras', 'sklearn', 'scikit-learn',
        'sql', 'spark', 'kafka', 'airflow', 'dbt', 'flink',
        'aws', 'gcp', 'azure', 'docker', 'kubernetes', 'terraform',
        'data science', 'reinforcement learning',
        'mlops', 'feature engineering', 'a/b testing',
        'embeddings', 'faiss', 'pinecone', 'weaviate', 'qdrant', 'milvus',
        'elasticsearch', 'opensearch', 'vector search', 'hybrid search',
        'rag', 'retrieval', 'reranking', 'semantic search',
        'lora', 'qlora', 'peft', 'fine-tuning',
        'ndcg', 'mrr', 'learning to rank',
        'sentence-transformers', 'bge', 'e5',
        'product management', 'analytics',
        'rest api', 'microservices',
    ]
    # Also extract from "required: x, y" pattern as fallback
    skill_patterns = [
        r'required[:\s]+([^.\n]+)',
        r'must.have[:\s]+([^.\n]+)',
    ]
    raw_skills = set()
    for sk in KNOWN_SKILLS:
        if sk in jd_lower:
            raw_skills.add(sk)
    for pat in skill_patterns:
        m = re.search(pat, jd_lower)
        if m:
            raw = m.group(1)
            for s in re.split(r'[,/]', raw):
                s = s.strip().strip(',').strip()
                # Skip leading "skills:" prefix artifacts
                s = re.sub(r'^skills?:\s*', '', s)
                if 1 < len(s) <= 30:
                    raw_skills.add(s)
    skills = sorted(raw_skills)

    # Extract years — handle ranges like "5-9 years" or "5–9 years"; take minimum
    years_range = re.search(r'(\d+)\s*[–\-–to]+\s*(\d+)\s*years?', jd_lower)
    years_single = re.search(r'(\d+)\+?\s*years?', jd_lower)
    if years_range:
        required_years = int(years_range.group(1))  # take minimum of range
    elif years_single:
        required_years = int(years_single.group(1))
    else:
        required_years = 3

    # Industries hinted at
    industry_keywords = {
        'fintech': ['fintech', 'finance', 'banking', 'payments'],
        'healthcare': ['healthcare', 'health', 'medical', 'pharma'],
        'e-commerce': ['ecommerce', 'retail', 'marketplace'],
        'saas': ['saas', 'b2b', 'enterprise software'],
        'ai/ml': ['ai', 'machine learning', 'llm', 'deep learning', 'nlp'],
    }
    industries = []
    for ind, kws in industry_keywords.items():
        if any(kw in jd_lower for kw in kws):
            industries.append(ind)

    # Preferred locations (from real JD)
    preferred_locations = ['pune', 'noida', 'hyderabad', 'mumbai', 'delhi', 'bangalore', 'bengaluru', 'chennai']
    # Disqualifier signals extracted from real JD text
    disqualify_companies = ['tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini']

    return {
        'raw': jd_text,
        'skills': [s for s in skills[:20] if len(s) > 1],
        'required_years': required_years,
        'industries': industries,
        'preferred_locations': preferred_locations,
        'disqualify_companies': disqualify_companies,
    }


# ──────────────────────────────────────────────
# Feature extractor (shared by heuristic + LTR)
# ──────────────────────────────────────────────

_PRODUCTION_VERBS = ['built', 'developed', 'shipped', 'deployed', 'implemented',
                     'designed', 'architected', 'led', 'owned', 'launched', 'created']
_AI_NOUNS = ['model', 'system', 'pipeline', 'retrieval', 'embedding', 'search',
             'ranker', 'recommender', 'classifier', 'llm', 'nlp', 'ml', 'ai',
             'vector', 'ranking', 'recommendation', 'inference']
_RETRIEVAL_KWS = ['retrieval', 'ranking', 'vector', 'faiss', 'embedding', 'reranking',
                  'bm25', 'elasticsearch', 'opensearch', 'semantic search', 'ndcg', 'mrr',
                  'pinecone', 'weaviate', 'qdrant', 'milvus', 'sentence-transformer', 'bge',
                  'learning to rank', 'lora', 'qlora', 'peft', 'rag', 'hybrid search']
_SENIORITY_KWS = ['senior', 'lead', 'principal', 'staff', 'architect', 'head', 'director']
_AI_COMPANY_KWS = ['ai', 'ml', 'data', 'tech', 'software', 'saas', 'platform']


def extract_features(candidate: dict, jd_req: dict) -> dict:
    """
    Extract a flat numerical feature dict for a candidate relative to a JD.

    This is the single source of truth for feature computation — used by both
    heuristic_score (for dimension scoring) and LTR training/inference.

    Features are split into two groups:
      - JD-match features: role-specific signals (skill overlap, years ratio, etc.)
      - Early-funnel behavioral features: platform signals that do NOT include
        interview_completion_rate or offer_acceptance_rate (those are the LTR label)
    """
    skills = candidate.get('skills') or []
    if isinstance(skills, str):
        try:
            skills = json.loads(skills.replace("'", '"'))
        except Exception:
            skills = [s.strip() for s in skills.split(',')]
    skills = [str(s).lower() for s in skills]

    years = float(candidate.get('years_of_experience') or 0)
    title = str(candidate.get('current_title') or '').lower()
    company = str(candidate.get('current_company') or '').lower()
    industry = str(candidate.get('current_industry') or '').lower()
    summary = str(candidate.get('summary') or '').lower()
    headline = str(candidate.get('headline') or '').lower()
    location_str = str(candidate.get('location') or '').lower()
    country_str = str(candidate.get('country') or '').lower()

    def _safe(val, lo=0.0, hi=1e9):
        try:
            v = float(val or 0)
        except (TypeError, ValueError):
            v = 0.0
        return max(lo, min(hi, v))

    required_skills  = jd_req.get('skills', [])
    required_years   = jd_req.get('required_years', 3)
    pref_locs        = jd_req.get('preferred_locations', [])
    pref_industries  = jd_req.get('industries', [])
    disqualify_cos   = [c.lower() for c in jd_req.get('disqualify_companies', [])]

    # ── JD-match signals ──────────────────────────────────────────
    skill_overlap = _skill_overlap(skills, required_skills)

    combined_text = ' '.join(skills) + ' ' + summary + ' ' + headline
    retrieval_kw_count = sum(1 for kw in _RETRIEVAL_KWS if kw in combined_text)

    required_years_clamped = max(required_years, 1)
    years_ratio = min(years / required_years_clamped, 2.0)
    years_raw   = years

    is_accepted  = int(any(kw in title for kw in _ACCEPTED_TITLE_KEYWORDS))
    is_rejected  = int(any(kw in title for kw in _REJECTED_TITLE_KEYWORDS))
    is_senior    = int(any(w in title for w in _SENIORITY_KWS))
    is_consulting = int(any(dc in company for dc in disqualify_cos))

    # Sentence-level production evidence
    sentences = re.split(r'[.!?;]', summary)
    prod_evidence = sum(
        1 for sent in sentences
        if any(v in sent for v in _PRODUCTION_VERBS) and any(n in sent for n in _AI_NOUNS)
    )
    prod_evidence = min(prod_evidence, 5)

    ind_match   = int(any(ind.lower() in industry for ind in pref_industries))
    is_product  = int(any(kw in company for kw in _AI_COMPANY_KWS) and not is_consulting)
    loc_score   = (1.0 if any(loc in location_str for loc in pref_locs)
                   else (0.7 if country_str == 'india' else 0.3))

    cskills = set(skills)
    required_set = set(s.lower() for s in required_skills)
    has_zero_required_skills = int(bool(required_set) and len(cskills & required_set) == 0)

    # ── Early-funnel behavioral signals (NOT the label) ───────────
    response_rate  = _safe(candidate.get('recruiter_response_rate'), 0, 1)
    saved_30d      = min(_safe(candidate.get('saved_by_recruiters_30d'), 0) / 20, 1.0)
    search_30d     = min(_safe(candidate.get('search_appearance_30d'), 0) / 500, 1.0)
    github_score   = min(_safe(candidate.get('github_activity_score'), 0) / 100, 1.0)
    open_to_work   = int(bool(candidate.get('open_to_work_flag')))
    willing_reloc  = int(bool(candidate.get('willing_to_relocate')))
    notice_days    = _safe(candidate.get('notice_period_days'), 0, 365)
    notice_norm    = 1.0 - min(notice_days / 180, 1.0)
    completeness   = _safe(candidate.get('profile_completeness_score'), 0, 100) / 100

    return {
        # JD-match
        'skill_overlap':            skill_overlap,
        'retrieval_kw_count':       min(retrieval_kw_count / 10, 1.0),
        'years_ratio':              years_ratio,
        'years_raw':                min(years_raw / 20, 1.0),
        'is_accepted_title':        is_accepted,
        'is_rejected_title':        is_rejected,
        'is_senior':                is_senior,
        'prod_evidence':            prod_evidence / 5.0,
        'is_product_company':       is_product,
        'is_consulting':            is_consulting,
        'industry_match':           ind_match,
        'location_score':           loc_score,
        'has_zero_required_skills': has_zero_required_skills,
        # Early-funnel behavioral
        'response_rate':            response_rate,
        'saved_30d_norm':           saved_30d,
        'search_30d_norm':          search_30d,
        'github_score':             github_score,
        'open_to_work':             open_to_work,
        'willing_relocate':         willing_reloc,
        'notice_norm':              notice_norm,
        'completeness':             completeness,
    }


def ltr_score(
    candidate: dict,
    jd_req: dict,
    ltr_artifact: dict,
    rrf_score: float = 0.0,
    bm25_score_norm: float = 0.0,
    dense_score: float = 0.0,
) -> float:
    """
    Score a candidate using the trained CatBoost LTR model.
    Retrieval scores are passed as features so the model can learn to weight
    BM25/dense/RRF signals alongside the profile-level features.
    Returns a score in roughly [0, 1].
    """
    import numpy as np
    feats = extract_features(candidate, jd_req)
    feats['rrf_score']        = rrf_score
    feats['bm25_score_norm']  = bm25_score_norm
    feats['dense_score']      = dense_score
    model = ltr_artifact['model']
    feature_names = ltr_artifact['feature_names']
    X = np.array([[feats.get(f, 0.0) for f in feature_names]])
    return float(model.predict(X)[0])


# ──────────────────────────────────────────────
# Heuristic judge
# ──────────────────────────────────────────────

def _skill_overlap(candidate_skills, required_skills) -> float:
    """
    Fraction of required skills covered by candidate, with synonym expansion.
    For each required skill, a match counts if the candidate has it directly
    or has any known synonym/alias of it.
    """
    if not required_skills:
        return 0.5
    cskills = set(s.lower() for s in (candidate_skills or []))
    rskills = [s.lower() for s in required_skills]
    if not rskills:
        return 0.5
    matched = 0
    for req in rskills:
        if req in cskills:
            matched += 1
            continue
        # Required skill is a canonical key — check if candidate has any of its aliases
        if any(alias in cskills for alias in SKILL_SYNONYMS.get(req, [])):
            matched += 1
            continue
        # Required skill might itself be an alias — check if candidate has the canonical
        for canonical, aliases in SKILL_SYNONYMS.items():
            if req in aliases and (canonical in cskills or any(a in cskills for a in aliases)):
                matched += 1
                break
    return matched / len(rskills)


def heuristic_score(candidate: dict, jd_req: dict) -> tuple[DimensionScores, str]:
    """
    Compute 6-dimension scores using extract_features() for raw signal extraction.
    interview_rate and offer_rate are extracted separately here because they are
    the LTR training label and must NOT enter extract_features().
    """
    f = extract_features(candidate, jd_req)

    def _safe(val, lo=0.0, hi=1e9):
        try:
            v = float(val or 0)
        except (TypeError, ValueError):
            v = 0.0
        return max(lo, min(hi, v))

    # These are end-of-funnel signals — used in heuristic scoring but NOT in LTR features
    interview_rate = _safe(candidate.get('interview_completion_rate'), 0, 1)
    offer_rate     = _safe(candidate.get('offer_acceptance_rate'), 0, 1)
    notice_days    = _safe(candidate.get('notice_period_days'), 0, 365)
    saved_30d_raw  = _safe(candidate.get('saved_by_recruiters_30d'), 0)

    required_skills  = jd_req.get('skills', [])
    required_years   = jd_req.get('required_years', 3)
    preferred_locs   = jd_req.get('preferred_locations', [])

    is_rejected  = bool(f['is_rejected_title'])
    is_accepted  = bool(f['is_accepted_title'])
    open_to_work = bool(f['open_to_work'])
    prod_count   = round(f['prod_evidence'] * 5)   # un-normalise for title_align logic

    # 1. Technical skill match (0-10)
    retrieval_bonus = f['retrieval_kw_count'] * 10 * 0.08
    skill_score = round(min(f['skill_overlap'] * 9 + retrieval_bonus, 10), 1)

    # 2. Career trajectory (0-10)
    if is_rejected and prod_count == 0:
        title_align = 0.15
    elif is_rejected:
        title_align = 0.35
    elif not is_accepted:
        title_align = 0.6
    else:
        title_align = 1.0

    prod_multiplier  = 1.0 + f['prod_evidence'] * 0.5
    seniority_bonus  = 1.0 if f['is_senior'] else 0.5
    product_boost    = 0.5 if not f['is_consulting'] else 0.0
    traj_score = round(min(
        (f['years_ratio'] * 4.0 + seniority_bonus * 2.5 + product_boost) * title_align * prod_multiplier, 10
    ), 1)

    # 3. Domain relevance (0-10)
    ind_mult    = 1.0 if f['industry_match'] else 0.4
    prod_domain = 1.0 if f['is_product_company'] else 0.0
    domain_score = round(min(ind_mult * 6 + prod_domain * 2 + f['location_score'] * 2, 10), 1)

    # 4. Behavioral engagement (0-10)
    resp_norm    = min(f['response_rate'] / 0.8, 1.0)
    int_norm     = min(interview_rate / 0.7, 1.0)
    inactive_pen = -2.0 if f['response_rate'] < 0.1 and not open_to_work else 0.0
    behavioral_score = round(max(
        resp_norm * 3 + int_norm * 3 + f['search_30d_norm'] * 2 + f['saved_30d_norm'] * 2 + inactive_pen, 0
    ), 1)

    # 5. Cultural fit (0-10)
    avail_bonus  = 2.0 if open_to_work else 0.0
    reloc_bonus  = 1.5 if f['willing_relocate'] else 0.0
    notice_bonus = 2.5 if notice_days <= 30 else (1.5 if notice_days <= 60 else (0.5 if notice_days <= 90 else 0.0))
    culture_score = round(min(avail_bonus + reloc_bonus + notice_bonus + f['completeness'] * 4.0, 10), 1)

    # 6. Bonus signals (0-10)
    offer_norm   = min(offer_rate / 0.8, 1.0)
    market_norm  = min(saved_30d_raw / 15, 1.0)
    bonus_score  = round(f['github_score'] * 5 + offer_norm * 3 + market_norm * 2, 1)

    def _clamp(v): return round(max(0.0, min(10.0, v)), 1)

    scores = DimensionScores(
        technical_skill_match=_clamp(skill_score),
        career_trajectory=_clamp(traj_score),
        domain_relevance=_clamp(domain_score),
        behavioral_engagement=_clamp(behavioral_score),
        cultural_fit=_clamp(culture_score),
        bonus_signals=_clamp(bonus_score),
    )

    # ── Reasoning ────────────────────────────────────────────────
    skills = candidate.get('skills') or []
    if isinstance(skills, str):
        try:
            skills = json.loads(skills.replace("'", '"'))
        except Exception:
            skills = [s.strip() for s in skills.split(',')]
    skills = [str(s).lower() for s in skills]

    title    = str(candidate.get('current_title') or '').lower()
    company  = str(candidate.get('current_company') or '')
    industry = str(candidate.get('current_industry') or '').lower()
    years    = float(candidate.get('years_of_experience') or 0)
    location_str = str(candidate.get('location') or '').lower()
    country_str  = str(candidate.get('country') or '').lower()

    matched = [s for s in skills if s in [r.lower() for r in required_skills]]
    missing = [r for r in required_skills if r.lower() not in set(skills)]

    reasoning_parts = []

    if matched:
        reasoning_parts.append(
            f"Matches {len(matched)}/{len(required_skills)} required skills"
            f" ({', '.join(matched[:4])}{'...' if len(matched) > 4 else ''})."
        )
    else:
        reasoning_parts.append(f"No direct skill overlap with required skills ({', '.join(required_skills[:3])}).")

    if missing:
        reasoning_parts.append(f"Missing: {', '.join(missing[:3])}{'...' if len(missing) > 3 else ''}.")

    exp_desc = f"{years:.1f} years ({title} at {company})"
    if years >= required_years * 1.5:
        reasoning_parts.append(f"Overqualified: {exp_desc} vs {required_years}+ required.")
    elif years >= required_years:
        reasoning_parts.append(f"Strong experience: {exp_desc} meets {required_years}+ year requirement.")
    else:
        reasoning_parts.append(f"Below required experience: {exp_desc}, needs {required_years}+.")

    if open_to_work:
        reasoning_parts.append("Actively open to new opportunities.")
    elif f['response_rate'] < 0.2:
        reasoning_parts.append("Appears inactive — low response rate and not open to work.")

    if notice_days <= 30:
        reasoning_parts.append(f"Available quickly ({int(notice_days)}-day notice).")
    elif notice_days > 90:
        reasoning_parts.append(f"Long notice period ({int(notice_days)} days) may slow hiring.")

    if f['github_score'] > 0.5:
        reasoning_parts.append(f"Strong open-source presence (GitHub score: {f['github_score']*100:.0f}/100).")

    if f['response_rate'] >= 0.7:
        reasoning_parts.append(f"Highly responsive to recruiters ({f['response_rate']:.0%}).")
    elif f['response_rate'] < 0.2:
        reasoning_parts.append(f"Very low recruiter responsiveness ({f['response_rate']:.0%}).")

    if f['industry_match']:
        reasoning_parts.append(f"Direct industry match ({industry}).")

    if any(loc in location_str for loc in preferred_locs):
        reasoning_parts.append(f"Location match ({candidate.get('location', '')}).")
    elif country_str == 'india':
        reasoning_parts.append(f"India-based ({candidate.get('location', '')}) — relocation may be needed.")

    if f['is_consulting']:
        reasoning_parts.append(f"NOTE: Currently at {company} — JD flags consulting-only backgrounds.")

    if is_rejected:
        if prod_count > 0:
            reasoning_parts.append(
                f"Career pivot signal: title is '{candidate.get('current_title','')}' but summary shows "
                f"{prod_count} production AI indicators — evaluated on content, not title alone."
            )
        else:
            reasoning_parts.append(
                f"Non-technical role ('{candidate.get('current_title','')}') with no production AI evidence "
                f"in summary — likely a keyword-listed profile rather than genuine AI background."
            )

    return scores, ' '.join(reasoning_parts)


# ──────────────────────────────────────────────
# Listwise LLM Ranker
# ──────────────────────────────────────────────

LISTWISE_SYSTEM = (
    "You are an expert technical recruiter. "
    "You evaluate and rank candidates fairly based on role fit, "
    "not surface keyword matching. You always output valid JSON."
)

LISTWISE_RANK_PROMPT = """Hiring for: {role_title} at {hiring_company}

Job Description:
{jd}

Below are {n} candidates. Rank ALL {n} by suitability for this role (most suitable = rank 1).

Focus on: technical depth with JD skills, career progression, domain fit, production evidence.

CANDIDATES:
{candidates}

Output ONLY valid JSON — just the ranked IDs, no reasoning here:
{{
  "ranking": [{{"rank": 1, "id": <index>}}, {{"rank": 2, "id": <index>}}, ...all {n}...]
}}"""

REASONING_PROMPT = """You are evaluating candidates FOR the role of "{role_title}" at {hiring_company}.

Job Requirements (summary):
{jd}

For each candidate below, write ONE specific sentence explaining fit for the {role_title} role at {hiring_company}.
Rules:
- Always refer to the TARGET role ("{role_title} at {hiring_company}"), NOT the candidate's current employer.
- Use the provided skill match data — mention matched skills, missing skills, or experience gap explicitly.
- Be concrete: "Matches 4/6 required skills including embeddings and RAG; missing lora and ndcg" beats "has ML experience".

{candidates}

Output ONLY valid JSON:
{{
  "explanations": [
    {{"id": <index>, "reasoning": "<one sentence: skill coverage + key gap or strength for the {role_title} role>"}},
    ...
  ]
}}"""


class ListwiseLLMRanker:
    """
    Ranks the top-N shortlisted candidates in a SINGLE LLM forward pass.

    Unlike pointwise scoring (one candidate per call), listwise ranking:
    - Sees all candidates simultaneously → reasons about relative merit
    - Single call → fast (15-20s for 50 candidates on GPU)
    - Output reasoning is per-candidate, not per-dimension
    - Zero training required — uses instruction-following zero-shot
    """

    def __init__(self, model_name: str = 'Qwen/Qwen2.5-3B-Instruct'):
        print(f"Loading listwise ranker: {model_name}")
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        self.model_name = model_name
        self.pipe = pipeline(
            'text-generation',
            model=model_name,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            max_new_tokens=2048,
            do_sample=False,
        )
        print("Listwise ranker ready.")

    @staticmethod
    def _candidate_summary(idx: int, cand: dict, required_skills: list | None = None) -> str:
        """Candidate summary for the prompt, including skill gap data when required_skills provided."""
        skills = cand.get('skills') or []
        if isinstance(skills, str):
            try:
                skills = json.loads(skills.replace("'", '"'))
            except Exception:
                skills = [s.strip() for s in skills.split(',')]
        skill_set = set(str(s).lower() for s in skills)
        top_skills = ', '.join(str(s) for s in skills[:6])
        summary_snippet = str(cand.get('summary') or '')[:120].replace('\n', ' ')

        base = (
            f"[{idx}] Current role: {cand.get('current_title', 'N/A')} @ "
            f"{cand.get('current_company', 'N/A')} "
            f"({cand.get('years_of_experience', 0):.0f} yrs exp, {cand.get('location', '')})\n"
            f"    Skills: {top_skills}\n"
            f"    Bio: {summary_snippet}"
        )

        if required_skills:
            req_lower = [r.lower() for r in required_skills]
            matched = [r for r in req_lower if r in skill_set]
            missing = [r for r in req_lower if r not in skill_set]
            gap_line = (
                f"    Required skill coverage: {len(matched)}/{len(req_lower)} matched"
                f" ({', '.join(matched[:4])}{'...' if len(matched) > 4 else ''})"
            )
            if missing:
                gap_line += f"; missing: {', '.join(missing[:4])}{'...' if len(missing) > 4 else ''}"
            base += f"\n{gap_line}"

        return base

    def _call_llm(self, messages: list, max_tokens: int = 1200) -> str:
        """Single LLM call, returns assistant reply string."""
        import copy
        raw = self.pipe(messages, max_new_tokens=max_tokens)[0]['generated_text']
        if isinstance(raw, list):
            return raw[-1].get('content', '')
        return str(raw)

    def rank(
        self,
        candidates: list[dict],
        jd_text: str,
        reason_top_n: int = 15,
        role_title: str = 'the open role',
        hiring_company: str = 'the hiring company',
        required_skills: list | None = None,
    ) -> list[dict]:
        """
        Two-pass listwise ranking:

        Pass 1 — Ranking:   all N candidates → ranked ID list (no reasoning needed,
                             compact output = better ranking quality within token budget)
        Pass 2 — Reasoning: top reason_top_n candidates → specific per-candidate
                             explanations grounded in skill gap data.

        Returns list sorted by listwise_rank ascending (rank 1 = best).
        """
        n = len(candidates)
        if n == 0:
            return []

        cand_summaries = [
            self._candidate_summary(i + 1, c, required_skills)
            for i, c in enumerate(candidates)
        ]
        cand_block = '\n\n'.join(cand_summaries)
        jd_short   = jd_text[:700]

        # ── Pass 1: Ranking ──
        rank_prompt = LISTWISE_RANK_PROMPT.format(
            jd=jd_short, n=n, candidates=cand_block,
            role_title=role_title, hiring_company=hiring_company,
        )
        rank_reply = self._call_llm(
            [{'role': 'system', 'content': LISTWISE_SYSTEM},
             {'role': 'user',   'content': rank_prompt}],
            max_tokens=n * 12 + 100,   # ~12 tokens per entry: {"rank":1,"id":3}
        )
        ranked = self._parse_ranking(rank_reply, candidates, n)

        # ── Pass 2: Reasoning for top reason_top_n ──
        top_results = sorted(ranked, key=lambda x: x['listwise_rank'])[:reason_top_n]
        top_indices = []
        for r in top_results:
            cid = r['candidate_id']
            for i, c in enumerate(candidates):
                if c.get('candidate_id') == cid:
                    top_indices.append(i + 1)   # 1-based
                    break

        top_cand_block = '\n\n'.join(
            cand_summaries[i - 1] for i in top_indices
        )
        rsn_prompt = REASONING_PROMPT.format(
            jd=jd_short,
            candidates=top_cand_block,
            role_title=role_title,
            hiring_company=hiring_company,
        )
        rsn_reply = self._call_llm(
            [{'role': 'system', 'content': LISTWISE_SYSTEM},
             {'role': 'user',   'content': rsn_prompt}],
            max_tokens=reason_top_n * 80 + 100,
        )
        reasoning_map = self._parse_reasoning(rsn_reply, top_indices, candidates)

        # Merge reasoning into ranked results
        for r in ranked:
            cid = r['candidate_id']
            r['reasoning'] = reasoning_map.get(cid, '')

        return sorted(ranked, key=lambda x: x['listwise_rank'])

    def _parse_reasoning(
        self, reply: str, top_indices: list[int], candidates: list[dict]
    ) -> dict:
        """Parse Pass-2 reasoning output → {candidate_id: reasoning_str}."""
        result = {}
        json_match = re.search(r'\{[\s\S]*"explanations"[\s\S]*\}', reply)
        if json_match:
            try:
                data = json.loads(json_match.group())
                for i, entry in enumerate(data.get('explanations', [])):
                    idx = int(entry.get('id', 0))
                    rsn = str(entry.get('reasoning', ''))
                    # Map numeric index back to candidate_id
                    if 1 <= idx <= len(candidates):
                        cid = candidates[idx - 1].get('candidate_id', '')
                        if cid:
                            result[cid] = rsn
                    # Fallback: use position in top_indices list
                    elif i < len(top_indices):
                        actual_idx = top_indices[i]
                        cid = candidates[actual_idx - 1].get('candidate_id', '')
                        if cid:
                            result[cid] = rsn
            except Exception:
                pass

        # Fallback: extract quoted sentences line by line
        if not result:
            lines = [l.strip() for l in reply.split('\n') if len(l.strip()) > 30]
            for i, (idx, line) in enumerate(zip(top_indices, lines)):
                cid = candidates[idx - 1].get('candidate_id', '')
                if cid:
                    result[cid] = line[:300]
        return result

    def _parse_ranking(self, reply: str, candidates: list[dict], n: int) -> list[dict]:
        """Parse JSON ranking output with robust fallback."""
        ranked = {}
        reasoning_map = {}

        # Try full JSON parse first
        json_match = re.search(r'\{[\s\S]*"ranking"[\s\S]*\}', reply)
        if json_match:
            try:
                data = json.loads(json_match.group())
                for entry in data.get('ranking', []):
                    idx  = int(entry.get('id', 0))
                    rank = int(entry.get('rank', 0))
                    rsn  = str(entry.get('reasoning', ''))
                    if 1 <= idx <= n and rank > 0:
                        ranked[idx] = rank
                        reasoning_map[idx] = rsn
            except Exception:
                pass

        # Fallback: scan for patterns like `"rank": 3, "id": 7`
        if not ranked:
            for m in re.finditer(r'"rank"\s*:\s*(\d+)[^}]*"id"\s*:\s*(\d+)', reply):
                rank, idx = int(m.group(1)), int(m.group(2))
                if 1 <= idx <= n and idx not in ranked:
                    ranked[idx] = rank

        # Fill missing with worst ranks
        used_ranks = set(ranked.values())
        next_rank  = n + 1
        for i in range(1, n + 1):
            if i not in ranked:
                while next_rank in used_ranks:
                    next_rank += 1
                ranked[i] = next_rank
                next_rank += 1

        # Convert to per-candidate output, score = 1 - (rank-1)/n → [0,1]
        results = []
        for i, cand in enumerate(candidates, start=1):
            r = ranked[i]
            results.append({
                'candidate_id':    cand.get('candidate_id', ''),
                'listwise_rank':   r,
                'listwise_score':  round(1.0 - (r - 1) / n, 4),
                'reasoning':       reasoning_map.get(i, ''),
            })

        results.sort(key=lambda x: x['listwise_rank'])
        return results


# ──────────────────────────────────────────────
# LLM judge (local transformers model)
# ──────────────────────────────────────────────

LLM_PROMPT_TEMPLATE = """You are an expert technical recruiter evaluating a candidate for a job opening.

Job Description:
{jd}

Candidate Profile:
- Name/Title: {title} at {company}
- Experience: {years} years
- Industry: {industry}
- Skills: {skills}
- Summary: {summary}
- Open to work: {open_to_work}
- GitHub activity score: {github_score}/100
- Recruiter response rate: {response_rate}
- Interview completion rate: {interview_rate}

Score this candidate on each dimension from 0 to 10 and provide a brief reasoning.
Return ONLY valid JSON in this exact format:
{{
  "technical_skill_match": <0-10>,
  "career_trajectory": <0-10>,
  "domain_relevance": <0-10>,
  "behavioral_engagement": <0-10>,
  "cultural_fit": <0-10>,
  "bonus_signals": <0-10>,
  "reasoning": "<2-3 sentence explanation of overall fit>"
}}"""


class LocalLLMJudge:
    """LLM judge using a local HuggingFace instruction-following model."""

    def __init__(self, model_name: str = 'Qwen/Qwen2.5-3B-Instruct'):
        print(f"Loading LLM judge model: {model_name}")
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        self.pipe = pipeline(
            'text-generation',
            model=model_name,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            max_new_tokens=512,
            do_sample=False,
        )
        print("LLM judge ready.")

    def score(self, candidate: dict, jd_req: dict) -> tuple[DimensionScores, str]:
        skills = candidate.get('skills') or []
        if isinstance(skills, str):
            try:
                skills = json.loads(skills.replace("'", '"'))
            except Exception:
                skills = [s.strip() for s in skills.split(',')]

        prompt = LLM_PROMPT_TEMPLATE.format(
            jd=jd_req['raw'][:600],
            title=candidate.get('current_title', 'N/A'),
            company=candidate.get('current_company', 'N/A'),
            years=candidate.get('years_of_experience', 0),
            industry=candidate.get('current_industry', 'N/A'),
            skills=', '.join(str(s) for s in skills[:15]),
            summary=(candidate.get('summary') or '')[:300],
            open_to_work=candidate.get('open_to_work_flag', False),
            github_score=candidate.get('github_activity_score', 0),
            response_rate=candidate.get('recruiter_response_rate', 0),
            interview_rate=candidate.get('interview_completion_rate', 0),
        )

        messages = [{'role': 'user', 'content': prompt}]
        output = self.pipe(messages)[0]['generated_text']

        # Extract assistant reply
        if isinstance(output, list):
            reply = output[-1].get('content', '')
        else:
            reply = str(output)

        # Parse JSON from reply
        json_match = re.search(r'\{[^{}]*\}', reply, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                scores = DimensionScores(
                    technical_skill_match=float(data.get('technical_skill_match', 5)),
                    career_trajectory=float(data.get('career_trajectory', 5)),
                    domain_relevance=float(data.get('domain_relevance', 5)),
                    behavioral_engagement=float(data.get('behavioral_engagement', 5)),
                    cultural_fit=float(data.get('cultural_fit', 5)),
                    bonus_signals=float(data.get('bonus_signals', 5)),
                )
                reasoning = data.get('reasoning', 'No reasoning provided.')
                return scores, reasoning
            except Exception:
                pass

        # Fallback to heuristic if parsing fails
        return heuristic_score(candidate, jd_req)


# ──────────────────────────────────────────────
# Composite score + penalty
# ──────────────────────────────────────────────

# Acceptable title keywords for an AI/ML engineering role
_ACCEPTED_TITLE_KEYWORDS = [
    'software engineer', 'software developer',
    'ml engineer', 'machine learning engineer', 'machine learning',
    'ai engineer', 'ai researcher', 'ai specialist', 'applied ai',
    'data scientist', 'data engineer', 'data analyst',
    'research engineer', 'research scientist', 'applied scientist',
    'nlp engineer', 'nlp scientist',
    'backend engineer', 'backend developer',
    'full stack', 'fullstack',
    'platform engineer', 'infrastructure engineer',
    'sre', 'devops engineer',
    'senior engineer', 'principal engineer', 'staff engineer',
    'architect',  # software/data/ml architect
    'computer scientist',
    'analytics engineer',
    'deep learning', 'computer vision engineer', 'cv engineer',
]

# Titles that are definitively non-technical for this role
_REJECTED_TITLE_KEYWORDS = [
    'hr ', 'human resources', 'recruiter', 'talent acquisition',
    'marketing', 'brand manager', 'content writer', 'copywriter', 'social media',
    'sales ', 'account manager', 'business development',
    'financial', 'accountant', 'finance manager',
    'operations manager', 'project coordinator',
    'civil engineer', 'mechanical engineer', 'electrical engineer',
    'chemical engineer', 'structural engineer',
    'supply chain', 'logistics', 'procurement',
    'ceo', 'coo', 'cfo', 'chief operating', 'chief financial',
    'legal', 'compliance officer',
    'administrative', 'office manager',
]


def compute_composite(scores: DimensionScores, weights: dict, candidate: dict, jd_req: dict, rrf_score: float = 0.0) -> float:
    """Weighted sum (0-100) with hard-requirement and disqualifier penalties.

    rrf_score: Reciprocal Rank Fusion score from Stage 1 retrieval.
    Candidates that ranked highly in BOTH dense and BM25 systems get a small
    bonus (up to +5 pts) because dual retrieval agreement is a strong relevance signal.
    """
    raw = sum(getattr(scores, dim) * w for dim, w in weights.items())
    composite = raw * 10  # scale to 0-100

    # RRF bonus: dual-retrieval agreement signal (rrf_score range ~0.003–0.033 → 0–5 pts)
    composite += min(rrf_score * 200, 5)

    skills = candidate.get('skills') or []
    if isinstance(skills, str):
        try:
            skills = json.loads(skills.replace("'", '"'))
        except Exception:
            skills = [s.strip() for s in skills.split(',')]
    cskills = set(str(s).lower() for s in skills)
    required = set(s.lower() for s in jd_req.get('skills', []))
    title = str(candidate.get('current_title') or '').lower()

    is_accepted = any(kw in title for kw in _ACCEPTED_TITLE_KEYWORDS)
    is_rejected = any(kw in title for kw in _REJECTED_TITLE_KEYWORDS)

    # Soft title penalty — not a hard disqualifier, career_trajectory already accounts for it
    # But add a moderate composite-level penalty to reflect JD guidance
    if is_rejected:
        composite -= 15  # non-technical career path (HR, Civil Eng, etc.) — further evaluated via summary
    elif not is_accepted:
        composite -= 8   # title doesn't match known technical pattern

    # Penalty: no required skill overlap at all → -15 pts
    if required and len(cskills & required) == 0:
        composite -= 15

    # Penalty: consulting-only career (JD explicit disqualifier) → -20 pts
    company = str(candidate.get('current_company') or '').lower()
    disqualify = [c.lower() for c in jd_req.get('disqualify_companies', [])]
    if any(dc in company for dc in disqualify):
        composite -= 20

    # Penalty: very long notice period → -5 pts
    try:
        notice = float(candidate.get('notice_period_days') or 30)
    except (TypeError, ValueError):
        notice = 30
    if notice > 90:
        composite -= 5

    return round(max(composite, 0), 2)


# ──────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────

def run_judge(
    reranked_csv: str,
    candidate_data_path: str,
    jd_text: str,
    output_csv: str,
    output_json: str,
    mode: str = 'heuristic',
    top_k: int = 20,
    role_type: str = 'ic',
    model_name: str = 'Qwen/Qwen2.5-3B-Instruct',
):
    # Load reranked candidates
    reranked = pd.read_csv(reranked_csv)
    top_candidates = reranked.head(top_k)

    # Load full candidate profiles
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from eda import load_data
    all_candidates = load_data()

    top_ids = set(top_candidates['candidate_id'].tolist())
    profiles = all_candidates[all_candidates['candidate_id'].isin(top_ids)]
    profile_map = {row['candidate_id']: row.to_dict() for _, row in profiles.iterrows()}

    # Parse JD
    jd_req = parse_jd(jd_text)
    print(f"JD skills extracted: {jd_req['skills'][:10]}")
    print(f"Required experience: {jd_req['required_years']}+ years")

    # Initialize judge
    if mode == 'llm':
        judge = LocalLLMJudge(model_name)
        score_fn = judge.score
    else:
        score_fn = heuristic_score

    weights = ROLE_WEIGHTS.get(role_type, IC_WEIGHTS)

    # Score each candidate
    results = []
    explanations = {}

    print(f"\nScoring top {top_k} candidates ({mode} mode)...")
    for i, row in top_candidates.iterrows():
        cid = row['candidate_id']
        candidate = profile_map.get(cid, {})
        if not candidate:
            print(f"  [SKIP] {cid} — profile not found")
            continue

        scores, reasoning = score_fn(candidate, jd_req)
        composite = compute_composite(scores, weights, candidate, jd_req)

        # Recommendation tier
        if composite >= 75:
            recommendation = 'strong_yes'
        elif composite >= 60:
            recommendation = 'yes'
        elif composite >= 45:
            recommendation = 'maybe'
        else:
            recommendation = 'no'

        results.append({
            'candidate_id': cid,
            'llm_composite_score': composite,
            'semantic_score': row.get('semantic_score', 0),
            'rerank_score': row.get('rerank_score', 0),
            'technical_skill_match': scores.technical_skill_match,
            'career_trajectory': scores.career_trajectory,
            'domain_relevance': scores.domain_relevance,
            'behavioral_engagement': scores.behavioral_engagement,
            'cultural_fit': scores.cultural_fit,
            'bonus_signals': scores.bonus_signals,
            'recommendation': recommendation,
            'reasoning': reasoning,
            # Profile fields for display
            'current_title': candidate.get('current_title', ''),
            'current_company': candidate.get('current_company', ''),
            'years_of_experience': candidate.get('years_of_experience', 0),
            'location': candidate.get('location', ''),
            'current_industry': candidate.get('current_industry', ''),
        })

        explanations[cid] = {
            'candidate_id': cid,
            'profile': {
                'title': candidate.get('current_title'),
                'company': candidate.get('current_company'),
                'years': candidate.get('years_of_experience'),
                'headline': candidate.get('headline'),
                'skills': candidate.get('skills'),
                'industry': candidate.get('current_industry'),
            },
            'scores': asdict(scores),
            'composite_score': composite,
            'recommendation': recommendation,
            'reasoning': reasoning,
            'weights_used': weights,
        }

    # Sort by composite score
    results_df = pd.DataFrame(results).sort_values('llm_composite_score', ascending=False)
    results_df.insert(0, 'final_rank', range(1, len(results_df) + 1))

    # Save outputs
    results_df.to_csv(output_csv, index=False)
    with open(output_json, 'w') as f:
        json.dump(explanations, f, indent=2, default=str)

    print(f"\n✓ Final ranking saved to {output_csv}")
    print(f"✓ Full explanations saved to {output_json}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"TOP 10 CANDIDATES — {role_type.upper()} ROLE")
    print(f"{'='*70}")
    print(f"{'Rank':<5} {'ID':<14} {'Score':>6} {'Rec':<12} {'Title':<30} {'Co'}")
    print(f"{'-'*70}")
    for _, r in results_df.head(10).iterrows():
        print(
            f"{int(r['final_rank']):<5} {r['candidate_id']:<14} "
            f"{r['llm_composite_score']:>6.1f} {r['recommendation']:<12} "
            f"{str(r['current_title'])[:28]:<30} {r['current_company']}"
        )

    print(f"\nScore distribution:")
    print(f"  Mean: {results_df['llm_composite_score'].mean():.1f}")
    print(f"  Max:  {results_df['llm_composite_score'].max():.1f}")
    print(f"  Min:  {results_df['llm_composite_score'].min():.1f}")
    print(f"\nRecommendations:")
    for rec, cnt in results_df['recommendation'].value_counts().items():
        print(f"  {rec}: {cnt}")

    return results_df


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='LLM-as-Judge candidate scoring')
    parser.add_argument('--reranked-csv', default='reranked_results.csv',
                        help='Input: reranked candidates CSV')
    parser.add_argument('--output-csv', default='final_ranked.csv',
                        help='Output: final ranked CSV with dimension scores')
    parser.add_argument('--output-json', default='final_explanations.json',
                        help='Output: full per-candidate explanations JSON')
    parser.add_argument('--mode', choices=['heuristic', 'llm'], default='heuristic',
                        help='Scoring mode (heuristic=fast/no-model, llm=local transformers model)')
    parser.add_argument('--model', default='Qwen/Qwen2.5-3B-Instruct',
                        help='HuggingFace model for --mode llm')
    parser.add_argument('--top-k', type=int, default=20,
                        help='Number of candidates to score (default: 20)')
    parser.add_argument('--role-type', choices=['ic', 'lead'], default='ic',
                        help='Role type for score weighting')
    parser.add_argument('--jd', type=str, default=None,
                        help='Job description text (or path to .txt file)')
    args = parser.parse_args()

    # Load JD
    if args.jd and Path(args.jd).exists():
        jd_text = Path(args.jd).read_text()
    elif args.jd:
        jd_text = args.jd
    else:
        # Default JD from dataset context
        jd_text = (
            "Senior AI/ML Engineer. Required skills: machine learning, python, pytorch, "
            "tensorflow, deep learning, llm, transformers, nlp. "
            "5+ years experience. Looking for candidates with strong open-source contributions "
            "and experience in AI/ML product development."
        )
        print("No --jd provided; using default JD.")

    run_judge(
        reranked_csv=args.reranked_csv,
        candidate_data_path='',
        jd_text=jd_text,
        output_csv=args.output_csv,
        output_json=args.output_json,
        mode=args.mode,
        top_k=args.top_k,
        role_type=args.role_type,
        model_name=args.model,
    )


if __name__ == '__main__':
    main()
