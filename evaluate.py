#!/usr/bin/env python3
"""
Pipeline evaluation using behavioral proxy labels.

Since there are no human-labeled ground truth labels, relevance is approximated
using end-of-funnel behavioral signals:
    label = interview_completion_rate × 0.6 + offer_acceptance_rate × 0.4

Four systems are compared on the same ~999-candidate pool from Stage 1 retrieval:
    1. BM25 baseline         — top-100 by BM25 score alone
    2. Dense baseline        — top-100 by dense cosine similarity alone
    3. Stage 1: RRF only     — top-100 by Reciprocal Rank Fusion (no judge)
    4. Full pipeline         — submission.csv (hybrid retrieval + judge scoring)

Metrics:
    NDCG@K      — ranking quality at cutoff K (standard IR metric)
    Spearman ρ  — rank order correlation with behavioral proxy
    Precision@K — fraction of top-K with label > 0.5
    Trap rate   — fraction of trap candidates (rejected title + no prod evidence) in top-10

Usage:
    python evaluate.py
    python evaluate.py --submission submission.csv --jd job_description.txt
"""

import sys
import re
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score as _sklearn_ndcg

sys.path.insert(0, '.')


# ─────────────────────────────────────────────────────────────
# Label + signal helpers
# ─────────────────────────────────────────────────────────────

def behavioral_label(candidate: dict) -> float:
    """
    End-of-funnel proxy for candidate quality.
    Same formula used as the LTR training label so evaluation is consistent.
    """
    def _safe(val):
        try:
            v = float(val or 0)
        except (TypeError, ValueError):
            v = 0.0
        return max(0.0, min(1.0, v))

    return _safe(candidate.get('interview_completion_rate')) * 0.6 + \
           _safe(candidate.get('offer_acceptance_rate'))     * 0.4


def is_trap(candidate: dict) -> bool:
    """
    True if the candidate has a non-technical title AND no sentence-level
    production AI evidence in their summary — the classic keyword-stuffing trap.
    """
    from llm_judge import _REJECTED_TITLE_KEYWORDS, _PRODUCTION_VERBS, _AI_NOUNS
    title   = str(candidate.get('current_title') or '').lower()
    summary = str(candidate.get('summary')       or '').lower()

    if not any(kw in title for kw in _REJECTED_TITLE_KEYWORDS):
        return False

    sentences = re.split(r'[.!?;]', summary)
    has_prod  = any(
        any(v in s for v in _PRODUCTION_VERBS) and any(n in s for n in _AI_NOUNS)
        for s in sentences
    )
    return not has_prod


# ─────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────

def ndcg_at(y_true_1d: np.ndarray, y_score_1d: np.ndarray, k: int) -> float:
    return float(_sklearn_ndcg([y_true_1d], [y_score_1d], k=k))


def compute_metrics(
    pool_cids:   list,
    true_labels: np.ndarray,
    sys_scores:  np.ndarray,
    profile_map: dict,
) -> dict:
    """
    Compute all metrics for one system over the full retrieval pool.

    pool_cids   : list of candidate_ids in pool order
    true_labels : behavioral label for each candidate (same order as pool_cids)
    sys_scores  : system score for each candidate (0.0 if not retrieved by this system)
    profile_map : {candidate_id: candidate_dict}
    """
    n = len(pool_cids)
    assert len(true_labels) == n and len(sys_scores) == n

    # Ranked order by system score (descending)
    ranked_idx   = np.argsort(sys_scores)[::-1]
    top100_idx   = ranked_idx[:100]
    top10_idx    = ranked_idx[:10]

    top100_cids  = [pool_cids[i] for i in top100_idx]
    top100_labels = true_labels[top100_idx]
    top100_scores = sys_scores[top100_idx]
    top10_labels  = true_labels[top10_idx]

    # NDCG over the full pool (not just top-100) for fair comparison
    m = {}
    for k in (10, 20, 100):
        m[f'ndcg@{k}'] = ndcg_at(true_labels, sys_scores, k)

    # Spearman ρ over top-100: does our rank order agree with behavioral rank?
    rho, pval = spearmanr(top100_labels, top100_scores)
    m['spearman_rho']  = float(rho)
    m['spearman_pval'] = float(pval)

    # Precision@K: fraction of top-K with behavioral label above 0.5
    m['precision@10']  = float((top10_labels  > 0.5).mean())
    m['precision@20']  = float((true_labels[ranked_idx[:20]] > 0.5).mean())

    # Mean behavioral label (quality of selected candidates)
    m['mean_label@10']  = float(top10_labels.mean())
    m['mean_label@100'] = float(top100_labels.mean())

    # Trap analysis: count trap candidates in top-10 and top-100
    trap_in_top10  = [cid for cid in [pool_cids[i] for i in top10_idx]
                      if is_trap(profile_map.get(cid, {}))]
    trap_in_top100 = [cid for cid in top100_cids
                      if is_trap(profile_map.get(cid, {}))]
    m['traps_in_top10']  = len(trap_in_top10)
    m['traps_in_top100'] = len(trap_in_top100)

    # Score discrimination: std dev of top-100 scores (higher = model spreads candidates apart)
    m['score_std'] = float(np.std(top100_scores))

    # Store top100 for profile display
    m['_top100_cids'] = top100_cids

    return m


# ─────────────────────────────────────────────────────────────
# Retrieval (reuses Stage 1 code)
# ─────────────────────────────────────────────────────────────

def get_retrieval_pool(jd_text: str, all_candidates: pd.DataFrame) -> pd.DataFrame:
    """Run Stage 1 hybrid retrieval and return the full pool with all scores."""
    from generate_submission import stage1_hybrid_retrieval
    return stage1_hybrid_retrieval(jd_text, all_candidates, fetch_per_system=500)


# ─────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────

def evaluate(jd_text: str, submission_csv: str, all_candidates: pd.DataFrame) -> dict:

    profile_map = {r['candidate_id']: r.to_dict() for _, r in all_candidates.iterrows()}

    # ── Stage 1: get the retrieval pool ───────────────────────
    print("\nRunning Stage 1 hybrid retrieval to build the evaluation pool...")
    retrieved = get_retrieval_pool(jd_text, all_candidates)
    pool_cids = retrieved['candidate_id'].tolist()
    print(f"Pool size: {len(pool_cids)} candidates")

    # ── Behavioral labels for the pool ────────────────────────
    true_labels = np.array([behavioral_label(profile_map.get(cid, {})) for cid in pool_cids])
    print(f"Label stats — mean: {true_labels.mean():.4f}  "
          f"std: {true_labels.std():.4f}  "
          f"fraction > 0.5: {(true_labels > 0.5).mean():.2%}")

    # ── Build score vectors for each system ───────────────────
    rec = retrieved.set_index('candidate_id')

    def pool_scores(col: str) -> np.ndarray:
        return np.array([float(rec.loc[cid, col]) if cid in rec.index else 0.0
                         for cid in pool_cids])

    systems = {
        'BM25 baseline':     pool_scores('bm25_score'),
        'Dense baseline':    pool_scores('dense_score'),
        'Stage 1 RRF only':  pool_scores('rrf_score'),
    }

    # Full pipeline: submission.csv scores (unranked candidates get 0.0)
    sub_path = Path(submission_csv)
    if sub_path.exists():
        sub_df = pd.read_csv(sub_path)
        sub_lookup = {row['candidate_id']: float(row['score']) for _, row in sub_df.iterrows()}
        systems['Full pipeline'] = np.array(
            [sub_lookup.get(cid, 0.0) for cid in pool_cids]
        )
    else:
        print(f"WARNING: {submission_csv} not found — skipping full pipeline comparison.")

    # ── Compute and display results ───────────────────────────
    all_results = {}
    rows = []

    for name, scores in systems.items():
        m = compute_metrics(pool_cids, true_labels, scores, profile_map)
        all_results[name] = m
        rows.append({
            'system':        name,
            'ndcg@10':       m['ndcg@10'],
            'ndcg@20':       m['ndcg@20'],
            'ndcg@100':      m['ndcg@100'],
            'spearman_rho':  m['spearman_rho'],
            'precision@10':  m['precision@10'],
            'mean_label@10': m['mean_label@10'],
            'traps@10':      m['traps_in_top10'],
            'traps@100':     m['traps_in_top100'],
            'score_std':     m['score_std'],
        })

    # ── Print table ────────────────────────────────────────────
    print("\n")
    col_w = 22
    hdr = (f"{'System':<{col_w}} {'NDCG@10':>8} {'NDCG@20':>8} {'NDCG@100':>9} "
           f"{'Spear ρ':>8} {'P@10':>6} {'μ@10':>6} {'Trap@10':>8} {'Trap@100':>9} {'StdDev':>7}")
    sep = "─" * len(hdr)
    print(sep)
    print(hdr)
    print(sep)

    for row in rows:
        delta_ndcg = ""
        print(
            f"{row['system']:<{col_w}} "
            f"{row['ndcg@10']:>8.4f} "
            f"{row['ndcg@20']:>8.4f} "
            f"{row['ndcg@100']:>9.4f} "
            f"{row['spearman_rho']:>8.4f} "
            f"{row['precision@10']:>6.2f} "
            f"{row['mean_label@10']:>6.4f} "
            f"{row['traps@10']:>8d} "
            f"{row['traps@100']:>9d} "
            f"{row['score_std']:>7.4f}"
        )

    print(sep)
    print()
    print("Legend:")
    print("  NDCG@K      ranking quality at cutoff K (vs. behaviorally-ideal ranking)")
    print("  Spear ρ     rank correlation with behavioral proxy order for top-100")
    print("  P@10        fraction of top-10 with behavioral label > 0.5 (high-quality hires)")
    print("  μ@10        mean behavioral label of top-10 (0=worst, 1=best)")
    print("  Trap@K      trap candidates (rejected title + no prod evidence) in top-K")
    print("  StdDev      score spread in top-100 (higher = better discrimination)")
    print()
    print("Proxy label: interview_completion_rate × 0.6 + offer_acceptance_rate × 0.4")
    print("(not ground truth — use deltas between systems, not absolute values)")

    # ── Delta analysis: improvement of full pipeline over BM25 baseline ───
    if 'Full pipeline' in all_results and 'BM25 baseline' in all_results:
        fp = all_results['Full pipeline']
        bm = all_results['BM25 baseline']
        print("\nDelta: Full pipeline vs. BM25 baseline")
        print(f"  NDCG@10  : {fp['ndcg@10']  - bm['ndcg@10']:+.4f}")
        print(f"  NDCG@100 : {fp['ndcg@100'] - bm['ndcg@100']:+.4f}")
        print(f"  Spear ρ  : {fp['spearman_rho'] - bm['spearman_rho']:+.4f}")
        print(f"  Trap@10  : {fp['traps_in_top10'] - bm['traps_in_top10']:+d}  (negative = fewer traps)")

    # ── Top-10 profile dump for full pipeline ─────────────────
    if 'Full pipeline' in all_results and sub_path.exists():
        print("\n\nTop 10 — Full pipeline:")
        print(f"{'Rk':<4} {'Label':>6}  {'Title':<35} {'Company':<22} {'Loc':<18} {'Yrs'}")
        print("─" * 95)
        for i, cid in enumerate(all_results['Full pipeline']['_top100_cids'][:10], 1):
            cand = profile_map.get(cid, {})
            lbl  = behavioral_label(cand)
            trap = " ⚠ TRAP" if is_trap(cand) else ""
            print(
                f"{i:<4} {lbl:>6.3f}  "
                f"{str(cand.get('current_title',''))[:33]:<35} "
                f"{str(cand.get('current_company',''))[:20]:<22} "
                f"{str(cand.get('location',''))[:16]:<18} "
                f"{cand.get('years_of_experience', 0):.1f}"
                f"{trap}"
            )

    return all_results


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Evaluate ranking pipeline')
    parser.add_argument('--jd',         default='job_description.txt', help='JD text file')
    parser.add_argument('--submission', default='submission.csv',      help='Full pipeline submission CSV')
    args = parser.parse_args()

    jd_path = Path(args.jd)
    if not jd_path.exists():
        print(f"JD not found: {jd_path}"); sys.exit(1)
    jd_text = jd_path.read_text()

    print("Loading candidate profiles...")
    from eda import load_data
    all_candidates = load_data()
    print(f"Loaded {len(all_candidates):,} candidates.")

    evaluate(jd_text, args.submission, all_candidates)


if __name__ == '__main__':
    main()
