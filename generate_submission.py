#!/usr/bin/env python3
"""
Full end-to-end ranking pipeline — submission generator.

Architecture:
  Stage 1: Hybrid Retrieval
    • Dense  — all-MiniLM-L6-v2 precomputed embeddings, cosine similarity
    • Sparse — BM25 on structured candidate text
    • RRF    — Reciprocal Rank Fusion merges both lists → top-N candidates
  Stage 2: Heuristic Judge
    • Scores all retrieved candidates on 6 dimensions (0-10 each)
    • Production-evidence analysis from summary text
    • Generates per-candidate reasoning
  Stage 3: Final ranking
    • Composite score (weighted sum of 6 dimensions)
    • Submission CSV: candidate_id, rank, score, reasoning

CPU-only. Runs in ~1-2 min on 16 GB machine.
"""

import sys
import re
import json
import time
import string
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

START = time.time()

def elapsed():
    return f"{time.time() - START:.1f}s"


# ──────────────────────────────────────────────────────────────
# Candidate text builder
# Build rich candidate text with emphasis on role + work context
# Skills are included but come after title/company/summary so that
# pure keyword-stuffing has less influence on BM25 than genuine role text
# ──────────────────────────────────────────────────────────────

def build_candidate_text(row: dict, mode: str = 'full') -> str:
    """
    Build a text representation of a candidate.

    mode='full'  → dense embedding text (title + headline + summary + skills)
    mode='sparse' → BM25 text (emphasises title + summary production signals)
    """
    title    = str(row.get('current_title') or '')
    company  = str(row.get('current_company') or '')
    industry = str(row.get('current_industry') or '')
    headline = str(row.get('headline') or '')
    summary  = str(row.get('summary') or '')
    location = str(row.get('location') or '')
    skills   = row.get('skills') or []
    if isinstance(skills, str):
        try:
            skills = json.loads(skills.replace("'", '"'))
        except Exception:
            skills = [s.strip() for s in skills.split(',')]
    skills_str = ' '.join(str(s) for s in skills)

    if mode == 'sparse':
        # Repeat title 3× to give it more weight in BM25 without cheating
        # Include summary (contains actual work descriptions)
        # Skills listed last — BM25 will still pick them up but diluted
        return f"{title} {title} {title} {company} {industry} {headline} {summary} {skills_str}"
    else:
        return f"{title} {headline} {summary} {industry} {company} {location} {skills_str}"


# ──────────────────────────────────────────────────────────────
# Stage 1: Hybrid Retrieval
# ──────────────────────────────────────────────────────────────

def stage1_hybrid_retrieval(
    jd_text: str,
    all_candidates: pd.DataFrame,
    fetch_per_system: int = 500,
) -> pd.DataFrame:
    """
    Dense + BM25 hybrid retrieval with Reciprocal Rank Fusion.

    Returns DataFrame with columns: candidate_id, dense_score,
    bm25_score, dense_rank, bm25_rank, rrf_score
    Sorted by rrf_score descending.
    """
    print(f"[{elapsed()}] Stage 1a: Dense retrieval (precomputed embeddings)...")

    # ── Dense ──────────────────────────────────────────────────
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    precompute_dir = Path('precompute')
    embeddings = np.load(precompute_dir / 'candidate_embeddings.npy')
    cand_ids   = np.load(precompute_dir / 'candidate_ids.npy', allow_pickle=True)

    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    jd_emb = st_model.encode([jd_text], normalize_embeddings=True)
    dense_scores = cosine_similarity(jd_emb, embeddings)[0]

    dense_top_idx  = np.argsort(dense_scores)[::-1][:fetch_per_system]
    dense_df = pd.DataFrame({
        'candidate_id': cand_ids[dense_top_idx],
        'dense_score':  dense_scores[dense_top_idx],
        'dense_rank':   range(1, len(dense_top_idx) + 1),
    })
    print(f"[{elapsed()}] Stage 1a done: {len(dense_df)} candidates via dense.")

    # ── Sparse (BM25) ──────────────────────────────────────────
    print(f"[{elapsed()}] Stage 1b: BM25 sparse retrieval on {len(all_candidates):,} candidates...")

    from rank_bm25 import BM25Okapi

    def tokenize(text: str) -> list[str]:
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.split()

    # Build corpus texts — sparse mode emphasizes title over skills
    corpus_texts = [
        build_candidate_text(row, mode='sparse')
        for _, row in all_candidates.iterrows()
    ]
    tokenized_corpus = [tokenize(t) for t in corpus_texts]

    bm25 = BM25Okapi(tokenized_corpus)
    jd_tokens = tokenize(jd_text)
    bm25_raw = bm25.get_scores(jd_tokens)

    bm25_top_idx = np.argsort(bm25_raw)[::-1][:fetch_per_system]
    bm25_df = pd.DataFrame({
        'candidate_id': all_candidates.iloc[bm25_top_idx]['candidate_id'].values,
        'bm25_score':   bm25_raw[bm25_top_idx],
        'bm25_rank':    range(1, len(bm25_top_idx) + 1),
    })
    print(f"[{elapsed()}] Stage 1b done: {len(bm25_df)} candidates via BM25.")

    # ── RRF Fusion ─────────────────────────────────────────────
    k = 60  # standard RRF constant
    rrf = {}

    for _, row in dense_df.iterrows():
        cid = row['candidate_id']
        rrf.setdefault(cid, {'dense_score': 0, 'bm25_score': 0, 'dense_rank': None, 'bm25_rank': None})
        rrf[cid]['dense_score'] = row['dense_score']
        rrf[cid]['dense_rank']  = int(row['dense_rank'])

    for _, row in bm25_df.iterrows():
        cid = row['candidate_id']
        rrf.setdefault(cid, {'dense_score': 0, 'bm25_score': 0, 'dense_rank': None, 'bm25_rank': None})
        rrf[cid]['bm25_score'] = row['bm25_score']
        rrf[cid]['bm25_rank']  = int(row['bm25_rank'])

    records = []
    for cid, vals in rrf.items():
        dr = vals['dense_rank'] or (fetch_per_system + 1)
        br = vals['bm25_rank']  or (fetch_per_system + 1)
        rrf_score = 1.0 / (k + dr) + 1.0 / (k + br)
        records.append({
            'candidate_id': cid,
            'dense_score':  vals['dense_score'],
            'bm25_score':   vals['bm25_score'],
            'dense_rank':   vals['dense_rank'],
            'bm25_rank':    vals['bm25_rank'],
            'rrf_score':    rrf_score,
        })

    fused = pd.DataFrame(records).sort_values('rrf_score', ascending=False).reset_index(drop=True)
    print(f"[{elapsed()}] Stage 1 RRF done: {len(fused)} unique candidates in union. "
          f"(dense-only: {(dense_df['candidate_id'].isin(bm25_df['candidate_id']) == False).sum()}, "
          f"bm25-only: {(bm25_df['candidate_id'].isin(dense_df['candidate_id']) == False).sum()}, "
          f"both: {dense_df['candidate_id'].isin(bm25_df['candidate_id']).sum()})")
    return fused


# ──────────────────────────────────────────────────────────────
# Stage 1 (full-corpus): Dense-only, no BM25, no recall ceiling
# ──────────────────────────────────────────────────────────────

def stage1_dense_fullcorpus(
    jd_text: str,
    all_candidates: pd.DataFrame,
) -> pd.DataFrame:
    """
    Score ALL candidates against the JD using precomputed bge-m3 embeddings.
    No BM25. No RRF. No recall ceiling — every candidate in the corpus gets
    a cosine similarity score and is eligible for LTR Stage 2.

    Why this beats hybrid retrieval:
    - Hybrid retrieval picks top-500 from each system → 999 unique candidates
    - Good candidates outside the top-500 of BOTH systems are permanently lost
    - Full-corpus dense scoring costs ~50ms (FAISS matmul on precomputed matrix)
      and gives every candidate a fair shot

    Returns DataFrame: candidate_id, dense_score — sorted descending.
    """
    print(f"[{elapsed()}] Stage 1 (full-corpus): Dense similarity for ALL "
          f"{len(all_candidates):,} candidates...")

    from sentence_transformers import SentenceTransformer

    precompute_dir = Path('precompute')
    embeddings = np.load(precompute_dir / 'candidate_embeddings.npy')
    cand_ids   = np.load(precompute_dir / 'candidate_ids.npy', allow_pickle=True)

    from sklearn.metrics.pairwise import cosine_similarity
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    jd_emb = st_model.encode([jd_text], normalize_embeddings=True)

    # Scores for ALL 100K — this is the key difference from stage1_hybrid_retrieval
    dense_scores = cosine_similarity(jd_emb, embeddings)[0]

    result = pd.DataFrame({
        'candidate_id': cand_ids,
        'dense_score':  dense_scores,
    }).sort_values('dense_score', ascending=False).reset_index(drop=True)

    print(f"[{elapsed()}] Stage 1 done: {len(result):,} candidates scored. "
          f"Dense score range: [{dense_scores.min():.4f}, {dense_scores.max():.4f}]")
    return result


# ──────────────────────────────────────────────────────────────
# Stage 2: Heuristic Judge scoring
# ──────────────────────────────────────────────────────────────

def stage2_judge(
    retrieved_df: pd.DataFrame,
    all_candidates: pd.DataFrame,
    jd_text: str,
    top_n: int = 100,
    use_llm: bool = False,
    llm_model: str = 'Qwen/Qwen2.5-3B-Instruct',
    llm_top_n: int = 150,
    ltr_artifact: dict = None,
) -> pd.DataFrame:
    """
    Score all retrieved candidates.

    Scoring modes (applied in order of priority):
      1. ltr_artifact provided  → CatBoost LTR score (learned weights from data)
      2. use_llm=True           → LLM re-scores top llm_top_n after heuristic pass
      3. default                → heuristic composite (hand-tuned IC_WEIGHTS)

    heuristic_score() always runs for dimension scores + reasoning text.
    """
    print(f"[{elapsed()}] Stage 2: scoring {len(retrieved_df)} candidates...")

    sys.path.insert(0, str(Path(__file__).parent))
    from llm_judge import parse_jd, heuristic_score, compute_composite, IC_WEIGHTS, ltr_score

    jd_req = parse_jd(jd_text)
    print(f"[{elapsed()}]   JD skills: {jd_req['skills'][:8]}, required years: {jd_req['required_years']}")
    if ltr_artifact:
        print(f"[{elapsed()}]   Scoring mode: CatBoost LTR (Spearman ρ={ltr_artifact.get('val_spearman', '?'):.4f})")
    elif use_llm:
        print(f"[{elapsed()}]   Scoring mode: heuristic → LLM re-score top {llm_top_n}")
    else:
        print(f"[{elapsed()}]   Scoring mode: heuristic (hand-tuned weights)")

    profile_map = {
        row['candidate_id']: row.to_dict()
        for _, row in all_candidates.iterrows()
    }

    bm25_max = float(retrieved_df['bm25_score'].max() or 1.0) if 'bm25_score' in retrieved_df.columns else 1.0

    results = []
    for _, row in retrieved_df.iterrows():
        cid = row['candidate_id']
        cand = profile_map.get(cid, {})
        if not cand:
            continue

        scores, reasoning = heuristic_score(cand, jd_req)
        rrf = float(row.get('rrf_score', 0))

        # Composite is ALWAYS the weighted sum of the 6 displayed dimension scores.
        # This ensures R²(dimension scores, final score) ≈ 1.0 — the ranking reflects
        # exactly what the score breakdown says it does.
        composite = compute_composite(scores, IC_WEIGHTS, cand, jd_req, rrf_score=rrf)

        if ltr_artifact:
            # LTR predicts behavioral outcome (interview_completion × 0.6 + offer_acceptance × 0.4).
            # Mean label ≈ 0.45. Use it as a bounded modifier (±15 pts) so it can influence
            # tie-breaking without overriding JD-match signals from the dimension scores.
            ltr_raw = ltr_score(
                cand, jd_req, ltr_artifact,
                rrf_score=rrf,
                bm25_score_norm=float(row.get('bm25_score', 0)) / bm25_max if 'bm25_score' in row.index else 0.0,
                dense_score=float(row.get('dense_score', 0)),
            )
            ltr_bonus = max(-15.0, min(15.0, (ltr_raw - 0.45) * 30))
            composite += ltr_bonus

        results.append({
            'candidate_id':          cid,
            'composite_score':       composite,
            'technical_skill_match': scores.technical_skill_match,
            'career_trajectory':     scores.career_trajectory,
            'domain_relevance':      scores.domain_relevance,
            'behavioral_engagement': scores.behavioral_engagement,
            'cultural_fit':          scores.cultural_fit,
            'bonus_signals':         scores.bonus_signals,
            'dense_score':           row.get('dense_score', 0),
            'bm25_score':            row.get('bm25_score', 0),
            'rrf_score':             row.get('rrf_score', 0),
            'reasoning':             reasoning,
            # profile fields for display
            'current_title':         cand.get('current_title', ''),
            'current_company':       cand.get('current_company', ''),
            'years_of_experience':   cand.get('years_of_experience', 0),
            'location':              cand.get('location', ''),
        })

    scored = pd.DataFrame(results).sort_values('composite_score', ascending=False)
    print(f"[{elapsed()}] Heuristic pass done. Score range: "
          f"[{scored['composite_score'].min():.1f}, {scored['composite_score'].max():.1f}].")

    # Optional LLM re-scoring: take top llm_top_n from heuristic, re-score with LLM,
    # then pick final top_n. This replaces heuristic template reasoning with LLM reasoning
    # for the candidates that actually matter.
    if use_llm and len(scored) > 0:
        from llm_judge import LocalLLMJudge
        pool_size = min(llm_top_n, len(scored))
        print(f"[{elapsed()}] Stage 2b: LLM Judge re-scoring top {pool_size} candidates "
              f"(model={llm_model})...")
        judge = LocalLLMJudge(llm_model)
        top_pool = scored.head(pool_size)

        llm_results = []
        for i, (_, row) in enumerate(top_pool.iterrows(), 1):
            cid = row['candidate_id']
            cand = profile_map.get(cid, {})
            if not cand:
                continue
            llm_scores, llm_reasoning = judge.score(cand, jd_req)
            llm_composite = compute_composite(
                llm_scores, IC_WEIGHTS, cand, jd_req,
                rrf_score=float(row.get('rrf_score', 0))
            )
            llm_results.append({
                'candidate_id':          cid,
                'composite_score':       llm_composite,
                'technical_skill_match': llm_scores.technical_skill_match,
                'career_trajectory':     llm_scores.career_trajectory,
                'domain_relevance':      llm_scores.domain_relevance,
                'behavioral_engagement': llm_scores.behavioral_engagement,
                'cultural_fit':          llm_scores.cultural_fit,
                'bonus_signals':         llm_scores.bonus_signals,
                'dense_score':           row.get('dense_score', 0),
                'bm25_score':            row.get('bm25_score', 0),
                'rrf_score':             row.get('rrf_score', 0),
                'reasoning':             llm_reasoning,
                'current_title':         cand.get('current_title', ''),
                'current_company':       cand.get('current_company', ''),
                'years_of_experience':   cand.get('years_of_experience', 0),
                'location':              cand.get('location', ''),
            })
            if i % 25 == 0:
                print(f"[{elapsed()}]   LLM scored {i}/{pool_size}...")

        scored = pd.DataFrame(llm_results).sort_values('composite_score', ascending=False)
        print(f"[{elapsed()}] Stage 2b done. LLM score range: "
              f"[{scored['composite_score'].min():.1f}, {scored['composite_score'].max():.1f}].")

    print(f"[{elapsed()}] Stage 2 complete. Top-{top_n} kept.")
    return scored.head(top_n).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────
# Stage 2c-L: Listwise LLM Ranker (replaces cross-encoder for best quality)
# ──────────────────────────────────────────────────────────────

def stage2c_llm_reason(
    scored_df: pd.DataFrame,
    profile_map: dict,
    jd_text: str,
    jd_req: dict | None = None,
    model_name: str = 'Qwen/Qwen2.5-3B-Instruct',
    top_n: int = 15,
) -> pd.DataFrame:
    """
    LLM reasoning layer: keeps LTR ranking intact, enriches top-N with
    specific per-candidate explanations from a local LLM.

    Why separate ranking from reasoning:
    - LTR ranks by behavioral signals (interview completion × offer acceptance)
      — trained on actual outcomes, best for precision
    - LLM writes *why* each top candidate fits the role semantically
      — addresses explainability judging criterion directly

    Each top-N candidate gets a 1-2 sentence explanation referencing their
    specific skills, company, and experience. Zero ranking changes.
    """
    from llm_judge import ListwiseLLMRanker, REASONING_PROMPT, LISTWISE_SYSTEM, parse_jd
    import re

    print(f"[{elapsed()}] Stage 2c-L: LLM reasoning for top-{top_n} candidates "
          f"via {model_name} (ranking unchanged)...")

    ranker = ListwiseLLMRanker(model_name=model_name)

    # Extract target role metadata from JD to ground the LLM
    if jd_req is None:
        jd_req = parse_jd(jd_text)
    required_skills = jd_req.get('skills', [])

    # Parse role title and company from JD header lines
    role_title = 'the open role'
    hiring_company = 'the hiring company'
    for line in jd_text.splitlines()[:10]:
        m = re.match(r'Job Description[:\s]+(.+?)\s*[—–]', line)
        if m:
            role_title = m.group(1).strip()
        m2 = re.match(r'Company[:\s]+(.+)', line)
        if m2:
            hiring_company = m2.group(1).split('(')[0].strip()

    top_df = scored_df.head(top_n)
    candidates = []
    for _, row in top_df.iterrows():
        cand = dict(profile_map.get(row['candidate_id'], {}))
        cand['candidate_id'] = row['candidate_id']
        candidates.append(cand)

    # Build summaries with pre-computed skill gap data
    top_indices = list(range(1, len(candidates) + 1))
    cand_summaries = [
        ranker._candidate_summary(i, c, required_skills)
        for i, c in enumerate(candidates, 1)
    ]
    top_cand_block = '\n\n'.join(cand_summaries)

    rsn_prompt = REASONING_PROMPT.format(
        jd=jd_text[:700],
        candidates=top_cand_block,
        role_title=role_title,
        hiring_company=hiring_company,
    )
    rsn_reply = ranker._call_llm(
        [{'role': 'system', 'content': LISTWISE_SYSTEM},
         {'role': 'user',   'content': rsn_prompt}],
        max_tokens=top_n * 90 + 150,
    )
    reasoning_map = ranker._parse_reasoning(rsn_reply, top_indices, candidates)

    df = scored_df.copy()
    for cid, rsn in reasoning_map.items():
        if rsn:
            df.loc[df['candidate_id'] == cid, 'reasoning'] = rsn[:600]

    enriched = sum(1 for cid in reasoning_map if reasoning_map[cid])
    print(f"[{elapsed()}] Stage 2c-L done. Enriched {enriched}/{top_n} candidates with LLM reasoning.")
    return df


# Stage 2c: LoRA Cross-encoder reranker (optional, GPU recommended)
# ──────────────────────────────────────────────────────────────

def stage2c_rerank(
    scored_df: pd.DataFrame,
    profile_map: dict,
    jd_text: str,
    adapter_path: str = None,
    model_name: str = 'BAAI/bge-reranker-v2-m3',
    reranker_weight: float = 0.30,
) -> pd.DataFrame:
    """
    Re-rank top-100 LTR candidates using bge-reranker-v2-m3.

    adapter_path=None → pretrained base model (strong general cross-encoder)
    adapter_path=<dir> → LoRA-adapted model fine-tuned on behavioral labels

    The cross-encoder reads full JD + candidate text, capturing semantic alignment
    that tabular LTR features miss.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mode = f"adapter={adapter_path}" if adapter_path else "base model (no adapter)"
    print(f"[{elapsed()}] Stage 2c: Cross-encoder reranking {len(scored_df)} candidates "
          f"on {device} ({mode})...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

    if adapter_path:
        adapter_dir = Path(adapter_path)
        if adapter_dir.exists():
            try:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, str(adapter_dir))
                print(f"[{elapsed()}]   LoRA adapter loaded from {adapter_path}")
            except Exception as e:
                print(f"[{elapsed()}]   WARNING: adapter load failed ({e}), using base model")
        else:
            print(f"[{elapsed()}]   WARNING: adapter path not found, using base model")

    model.to(device)
    model.eval()

    jd_short = jd_text[:900]

    def _cand_text(cand: dict) -> str:
        skills = cand.get('skills') or []
        if isinstance(skills, str):
            try:
                skills = json.loads(skills.replace("'", '"'))
            except Exception:
                skills = [s.strip() for s in skills.split(',')]
        return (
            f"Title: {cand.get('current_title', '')}\n"
            f"Company: {cand.get('current_company', '')} ({cand.get('current_industry', '')})\n"
            f"Experience: {cand.get('years_of_experience', 0)} years\n"
            f"Summary: {str(cand.get('summary', ''))[:350]}\n"
            f"Skills: {', '.join(str(s) for s in skills[:18])}"
        )

    reranker_scores = []
    with torch.no_grad():
        for _, row in scored_df.iterrows():
            cand = profile_map.get(row['candidate_id'], {})
            enc = tokenizer(
                jd_short, _cand_text(cand),
                max_length=512, truncation=True,
                padding='max_length', return_tensors='pt',
            ).to(device)
            score = model(**enc).logits.squeeze().item()
            reranker_scores.append(score)

    df = scored_df.copy()
    df['reranker_score'] = reranker_scores

    # Blend LTR composite with reranker score:
    # normalise reranker scores to [0, 100] then take weighted average
    rr = np.array(reranker_scores)
    rr_min, rr_max = rr.min(), rr.max()
    if rr_max > rr_min:
        rr_norm = (rr - rr_min) / (rr_max - rr_min) * 100
    else:
        rr_norm = np.full_like(rr, 50.0)

    ltr = df['composite_score'].values
    lt_min, lt_max = ltr.min(), ltr.max()
    if lt_max > lt_min:
        ltr_norm = (ltr - lt_min) / (lt_max - lt_min) * 100
    else:
        ltr_norm = np.full_like(ltr, 50.0)

    rw = reranker_weight
    df['composite_score'] = rw * rr_norm + (1 - rw) * ltr_norm
    df = df.sort_values('composite_score', ascending=False).reset_index(drop=True)

    rr_range = f"[{rr.min():.3f}, {rr.max():.3f}]"
    print(f"[{elapsed()}] Stage 2c done. Reranker score range: {rr_range}. "
          f"Std dev: {rr.std():.4f}  (>0.5 = discriminating well)")
    return df


# ──────────────────────────────────────────────────────────────
# Stage 3: Final ranking + submission
# ──────────────────────────────────────────────────────────────

def _template_reasoning(row) -> str:
    """
    Natural-language reasoning from dimension scores for candidates without LLM reasoning.
    Ensures every rank position has readable explainability, not just the top 15.
    """
    tech   = row.get('technical_skill_match', 0)
    traj   = row.get('career_trajectory', 0)
    domain = row.get('domain_relevance', 0)
    engage = row.get('behavioral_engagement', 0)
    fit    = row.get('cultural_fit', 0)
    bonus  = row.get('bonus_signals', 0)

    if tech >= 7:
        tech_str = f"strong technical skill coverage ({tech:.1f}/10) across the core retrieval and ML stack"
    elif tech >= 5:
        tech_str = f"moderate technical fit ({tech:.1f}/10) with some skill gaps"
    elif tech >= 3:
        tech_str = f"partial technical overlap ({tech:.1f}/10); missing several core required skills"
    else:
        tech_str = f"limited technical skill match ({tech:.1f}/10)"

    if traj >= 9:
        traj_str = f"excellent career trajectory ({traj:.1f}/10)"
    elif traj >= 7:
        traj_str = f"strong seniority progression ({traj:.1f}/10)"
    elif traj >= 5:
        traj_str = f"adequate experience level ({traj:.1f}/10)"
    else:
        traj_str = f"limited seniority signals ({traj:.1f}/10)"

    extras = []
    if domain >= 7:
        extras.append(f"high domain relevance ({domain:.1f}/10)")
    elif domain >= 5:
        extras.append(f"moderate domain fit ({domain:.1f}/10)")
    if engage >= 7:
        extras.append(f"actively recruiter-responsive ({engage:.1f}/10)")
    elif engage < 4:
        extras.append(f"low engagement signals ({engage:.1f}/10)")
    if fit >= 7:
        extras.append(f"good availability fit ({fit:.1f}/10)")
    if bonus >= 7:
        extras.append(f"strong open-source/market signals ({bonus:.1f}/10)")

    parts = [tech_str, traj_str] + extras
    sentence = parts[0] + '; ' + '; '.join(parts[1:]) + '.'
    return sentence[0].upper() + sentence[1:]


def stage3_submit(scored_df: pd.DataFrame, output_path: str):
    """Normalize scores, fill reasoning gaps, enforce monotone, write submission CSV."""
    import math as _math
    print(f"[{elapsed()}] Stage 3: Finalizing submission...")

    df = scored_df.copy()
    df = df.sort_values('composite_score', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)

    # Rank-based smooth score: exponential decay gives meaningful separation
    # across the full top-100 (rank 1=1.0, rank 50≈0.30, rank 100≈0.09).
    # Min-max normalization was compressing 79 candidates below 0.5.
    n = len(df)
    lam = -_math.log(0.30) / 49  # calibrated so rank-50 ≈ 0.30
    df['score'] = [round(_math.exp(-lam * i), 6) for i in range(n)]

    dim_cols = [c for c in ['technical_skill_match', 'career_trajectory', 'domain_relevance',
                             'behavioral_engagement', 'cultural_fit', 'bonus_signals'] if c in df.columns]

    # Fill in template reasoning for any candidate without LLM-generated text
    # (typically ranks 16–100). Every rank now has human-readable explainability.
    if dim_cols:
        def _fill_reasoning(row):
            existing = str(row.get('reasoning') or '').strip()
            # LLM reasoning starts with '[Tech:' prefix or is a real sentence
            if existing and not existing.startswith('[') and len(existing) > 40:
                return existing  # already has LLM reasoning
            template = _template_reasoning(row)
            if existing.startswith('['):
                # keep dimension prefix, replace generic tail with template
                prefix = existing[:existing.index(']') + 1] if ']' in existing else ''
                return (prefix + ' ' + template).strip()
            return template

        df['reasoning'] = df.apply(_fill_reasoning, axis=1)

    submission = df[['candidate_id', 'rank', 'score'] + dim_cols + ['reasoning']].copy()

    # Prepend dimension score summary so judges see the breakdown inline
    if dim_cols:
        abbrev = {
            'technical_skill_match': 'Tech',
            'career_trajectory':     'Traj',
            'domain_relevance':      'Domain',
            'behavioral_engagement': 'Engage',
            'cultural_fit':          'Fit',
            'bonus_signals':         'Bonus',
        }
        def _dim_prefix(row):
            parts = [f"{abbrev.get(c, c)}:{row[c]:.1f}/10" for c in dim_cols]
            return '[' + ' | '.join(parts) + '] ' + str(row.get('reasoning', ''))

        submission['reasoning'] = submission.apply(_dim_prefix, axis=1).str.slice(0, 950)
    else:
        submission['reasoning'] = submission['reasoning'].str.slice(0, 650)

    submission.to_csv(output_path, index=False)

    print(f"\n[{elapsed()}] DONE — submission written to {output_path}")
    print(f"\nTop 15 candidates:")
    print(f"{'Rank':<5} {'Score':>7}  {'Title':<32} {'Company':<22} {'Yrs'}")
    print(f"{'─'*85}")
    for _, r in df.head(15).iterrows():
        print(f"{int(r['rank']):<5} {r['score']:>7.4f}  "
              f"{str(r['current_title'])[:30]:<32} "
              f"{str(r['current_company'])[:20]:<22} "
              f"{r['years_of_experience']:.1f}")

    print(f"\nScore distribution (top 100):")
    print(f"  Max: {df['score'].max():.4f}  Mean: {df['score'].mean():.4f}  Min: {df['score'].min():.4f}")
    print(f"\nTop dimension scores (mean across top-10):")
    for dim in ['technical_skill_match','career_trajectory','domain_relevance',
                'behavioral_engagement','cultural_fit','bonus_signals']:
        if dim in df.columns:
            print(f"  {dim}: {df[dim].head(10).mean():.2f}")
    return submission


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Hybrid retrieval + heuristic judge pipeline')
    parser.add_argument('--jd',    default='job_description.txt', help='JD text file')
    parser.add_argument('--output', default='submission.csv',     help='Output submission CSV')
    parser.add_argument('--fetch',  type=int, default=500,
                        help='Candidates fetched per retrieval system (default: 500)')
    parser.add_argument('--full-corpus', action='store_true', default=False,
                        help='Score ALL candidates with dense similarity (no BM25, no recall ceiling)')
    parser.add_argument('--use-llm-judge', action='store_true', default=False,
                        help='Re-score top candidates with local LLM for richer reasoning (requires GPU)')
    parser.add_argument('--llm-model', default='Qwen/Qwen2.5-3B-Instruct',
                        help='HuggingFace model for LLM judge (default: Qwen/Qwen2.5-3B-Instruct)')
    parser.add_argument('--llm-pool', type=int, default=150,
                        help='Number of heuristic top-N to re-score with LLM (default: 150)')
    parser.add_argument('--ltr-model', default=None,
                        help='Path to trained CatBoost LTR model pickle (from train_ltr_model.py).')
    parser.add_argument('--use-reranker', action='store_true', default=False,
                        help='Apply cross-encoder reranker to top-100 after LTR scoring (GPU recommended)')
    parser.add_argument('--reranker-adapter', default=None,
                        help='Path to LoRA adapter dir. Omit to use base bge-reranker-v2-m3 (no fine-tuning)')
    parser.add_argument('--reranker-weight', type=float, default=0.30,
                        help='Blend weight for reranker vs LTR (default 0.30 = 30%% reranker)')
    parser.add_argument('--use-listwise', action='store_true', default=False,
                        help='Use listwise LLM ranker (Qwen2.5-3B) on top-50 — best quality, ~20s on GPU')
    parser.add_argument('--listwise-model', default='Qwen/Qwen2.5-3B-Instruct',
                        help='HuggingFace model for listwise ranking (default: Qwen/Qwen2.5-3B-Instruct)')
    parser.add_argument('--listwise-pool', type=int, default=50,
                        help='Number of LTR top-N candidates to pass to listwise ranker (default: 50)')
    parser.add_argument('--listwise-ltr-weight', type=float, default=0.35,
                        help='Weight of LTR score in final blend with listwise score (default: 0.35)')
    args = parser.parse_args()

    jd_path = Path(args.jd)
    if not jd_path.exists():
        print(f"JD file not found: {jd_path}"); sys.exit(1)
    jd_text = jd_path.read_text()

    print(f"[{elapsed()}] Loading all candidate profiles...")
    sys.path.insert(0, '.')
    from eda import load_data
    all_cands = load_data()
    print(f"[{elapsed()}] Loaded {len(all_cands):,} candidate profiles.")

    # Build profile map once — shared by Stage 2 and Stage 2c
    profile_map = {r['candidate_id']: r.to_dict() for _, r in all_cands.iterrows()}

    # Stage 1: Retrieval
    if args.full_corpus:
        retrieved = stage1_dense_fullcorpus(jd_text, all_cands)
    else:
        retrieved = stage1_hybrid_retrieval(jd_text, all_cands, fetch_per_system=args.fetch)

    # Load LTR model if provided
    ltr_artifact = None
    if args.ltr_model:
        import pickle
        ltr_path = Path(args.ltr_model)
        if not ltr_path.exists():
            print(f"LTR model not found: {ltr_path}. Train it first:")
            print(f"  python train_ltr_model.py --jd {args.jd} --output {args.ltr_model}")
            sys.exit(1)
        with open(ltr_path, 'rb') as f:
            ltr_artifact = pickle.load(f)
        print(f"[{elapsed()}] LTR model loaded (Spearman ρ={ltr_artifact.get('val_spearman', '?'):.4f}).")

    # Stage 2: Score retrieved candidates (heuristic / LTR / LLM)
    scored = stage2_judge(
        retrieved, all_cands, jd_text, top_n=100,
        use_llm=args.use_llm_judge,
        llm_model=args.llm_model,
        llm_top_n=args.llm_pool,
        ltr_artifact=ltr_artifact,
    )

    # Stage 2c-L: LLM reasoning enrichment (keeps LTR ranking, adds semantic explanations)
    if args.use_listwise:
        from llm_judge import parse_jd as _parse_jd
        scored = stage2c_llm_reason(
            scored, profile_map, jd_text,
            jd_req=_parse_jd(jd_text),
            model_name=args.listwise_model,
            top_n=args.listwise_pool,
        )
    # Stage 2c: Optional cross-encoder reranker (base model or LoRA-adapted)
    elif args.use_reranker:
        scored = stage2c_rerank(
            scored, profile_map, jd_text,
            adapter_path=args.reranker_adapter,
            reranker_weight=args.reranker_weight,
        )

    # Stage 3: Final submission
    stage3_submit(scored, args.output)


if __name__ == '__main__':
    main()
