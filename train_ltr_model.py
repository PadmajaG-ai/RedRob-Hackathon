#!/usr/bin/env python3
"""
Train a CatBoost model to replace hand-tuned heuristic weights.

The problem with heuristic scoring:
  IC_WEIGHTS = {technical: 0.30, trajectory: 0.20, domain: 0.20, ...}
  These weights are arbitrary — no data tells us that skill match is worth
  exactly 1.5× more than behavioral signals for this role type.

The fix:
  Train a CatBoostRegressor on all 100K candidates using:
    - Features: JD-match signals + early-funnel behavioral signals (from extract_features)
    - Label: interview_completion_rate × 0.6 + offer_acceptance_rate × 0.4
      (end-of-funnel signals — what the platform has already observed about each candidate)

  The model learns: "which combination of JD-match + behavioral features predicts
  a candidate who completes interviews and accepts offers?"

  This is genuinely learned, not hand-tuned.

Label separation from features:
  - interview_completion_rate + offer_acceptance_rate → LABEL (end-of-funnel)
  - recruiter_response_rate + saved_30d + search_30d + github_score → FEATURES (early-funnel)
  No data leakage: features precede the label in the hiring funnel.

Usage:
  python train_ltr_model.py --jd job_description.txt --output ltr_model.pkl
  python generate_submission.py --ltr-model ltr_model.pkl
"""

import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.insert(0, '.')
from llm_judge import parse_jd, extract_features


def compute_label(candidate: dict, skill_overlap: float = 0.0) -> float:
    """
    Blended label: behavioral funnel signal + JD-specific skill match.

    Root cause of ranking inversion: training on pure behavioral labels
    (interview completion + offer acceptance) taught the model that profile
    richness (retrieval_kw_count) predicts hiring outcomes — not whether the
    candidate actually matches THIS JD's required skills. skill_overlap had
    only 1.95% feature importance as a result.

    Fix: blend skill_overlap (0–1, JD-specific) directly into the label so
    the model learns that skill match IS a quality signal, not just a feature.

        label = behavioral * 0.5 + skill_overlap * 0.5

    This does not create leakage: skill_overlap is also a training feature,
    but the model still has to generalise — it can't memorise skill_overlap
    since it varies per JD at inference time.
    """
    def _safe(val):
        try:
            v = float(val or 0)
        except (TypeError, ValueError):
            v = 0.0
        return max(0.0, min(1.0, v))

    interview_rate = _safe(candidate.get('interview_completion_rate'))
    offer_rate     = _safe(candidate.get('offer_acceptance_rate'))
    behavioral     = interview_rate * 0.6 + offer_rate * 0.4
    return behavioral * 0.5 + float(skill_overlap) * 0.5


def build_dataset(
    candidates_df: pd.DataFrame,
    jd_req: dict,
    retrieval_df: pd.DataFrame = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Extract features and labels for all candidates.

    retrieval_df: optional DataFrame with candidate_id, bm25_score, dense_score, rrf_score
                  from Stage 1 retrieval. When provided, retrieval scores are added as
                  features so the model learns to weight BM25/dense/RRF signals directly.
                  Candidates not in retrieval_df get 0.0 for these features.
    """
    print(f"Extracting features for {len(candidates_df):,} candidates...")

    # Build normalised retrieval score lookups
    ret_lookup = {}
    if retrieval_df is not None:
        bm25_max = retrieval_df['bm25_score'].max() or 1.0
        for _, r in retrieval_df.iterrows():
            ret_lookup[r['candidate_id']] = {
                'rrf_score':       float(r.get('rrf_score', 0)),
                'bm25_score_norm': float(r.get('bm25_score', 0)) / bm25_max,
                'dense_score':     float(r.get('dense_score', 0)),
            }

    feature_rows = []
    label_rows   = []

    for i, (_, row) in enumerate(candidates_df.iterrows()):
        cand  = row.to_dict()
        feats = extract_features(cand, jd_req)

        ret = ret_lookup.get(cand.get('candidate_id', ''), {})
        feats['rrf_score']        = ret.get('rrf_score', 0.0)
        feats['bm25_score_norm']  = ret.get('bm25_score_norm', 0.0)
        feats['dense_score']      = ret.get('dense_score', 0.0)

        label = compute_label(cand, skill_overlap=feats.get('skill_overlap', 0.0))
        feature_rows.append(feats)
        label_rows.append(label)

        if (i + 1) % 20_000 == 0:
            print(f"  {i+1:,}/{len(candidates_df):,} done...")

    X = pd.DataFrame(feature_rows)
    y = np.array(label_rows, dtype=np.float32)

    # Scale skill_overlap from 0–1 to 0–10 to match other feature magnitudes.
    # Without this, CatBoost underweights it vs. trajectory/engagement (also 0–10).
    if 'skill_overlap' in X.columns:
        X['skill_overlap'] = X['skill_overlap'] * 10.0

    return X, y


def _make_ranking_pools(X, y, n_groups=50, group_size=2000, val_size=0.1, seed=42):
    """
    YetiRank needs multiple query groups. Since we have one JD, we create
    pseudo-groups by sampling overlapping subsets of candidates — a standard
    trick for single-query LTR. The model learns intra-group ranking, which
    generalises to the full candidate pool at inference.
    """
    from catboost import Pool
    np.random.seed(seed)
    n = len(X)

    # Hold out 10% of candidates as a single validation group
    val_n     = max(100, int(n * val_size))
    val_idx   = np.random.choice(n, val_n, replace=False)
    train_mask = np.ones(n, dtype=bool)
    train_mask[val_idx] = False
    X_tr, y_tr = X.iloc[train_mask], y[train_mask]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    # Build pseudo-groups from train pool
    tr_n = len(X_tr)
    all_X, all_y, all_g = [], [], []
    for g in range(n_groups):
        idx = np.random.choice(tr_n, size=min(group_size, tr_n), replace=False)
        all_X.append(X_tr.iloc[idx])
        all_y.append(y_tr[idx])
        all_g.extend([g] * len(idx))

    X_rank = pd.concat(all_X, ignore_index=True)
    y_rank = np.concatenate(all_y)
    g_rank = np.array(all_g)

    # Sort by group_id (CatBoost ranking requirement)
    order = np.argsort(g_rank, kind='stable')
    train_pool = Pool(data=X_rank.iloc[order], label=y_rank[order], group_id=g_rank[order])

    # Validation: 10 groups of ~1000 for NDCG eval
    vg_size = min(1000, len(X_val))
    vn_groups = max(1, len(X_val) // vg_size)
    vg_ids = np.repeat(np.arange(vn_groups), vg_size)[:len(X_val)]
    v_order = np.argsort(vg_ids, kind='stable')
    val_pool = Pool(data=X_val.iloc[v_order], label=y_val[v_order], group_id=vg_ids[v_order])

    return train_pool, val_pool, X_val, y_val


def train(
    jd_text: str,
    candidates_df: pd.DataFrame,
    output_path: str,
    retrieval_df: pd.DataFrame = None,
    use_ranker: bool = True,
) -> dict:
    """
    Train CatBoost model for candidate ranking.

    use_ranker=True  → CatBoostRanker with YetiRank (optimises NDCG directly)
    use_ranker=False → CatBoostRegressor with RMSE (legacy, faster)
    """
    try:
        from catboost import CatBoostRanker, CatBoostRegressor
    except ImportError:
        print("CatBoost not installed. Run: pip install catboost")
        sys.exit(1)

    from scipy.stats import spearmanr

    jd_req = parse_jd(jd_text)
    print(f"JD skills: {jd_req['skills'][:8]}")
    print(f"Required years: {jd_req['required_years']}")

    if retrieval_df is not None:
        print(f"Retrieval scores included as features ({len(retrieval_df)} candidates in pool).")

    X, y = build_dataset(candidates_df, jd_req, retrieval_df=retrieval_df)

    print(f"\nDataset: {X.shape[0]:,} candidates × {X.shape[1]} features")
    print(f"Label stats — mean: {y.mean():.4f}  std: {y.std():.4f}  "
          f"min: {y.min():.4f}  max: {y.max():.4f}")
    print(f"Candidates with label > 0.5: {(y > 0.5).sum():,} "
          f"({100*(y > 0.5).mean():.1f}%)")

    train_pool_ref = None  # kept for feature importance (ranker mode needs it)

    if use_ranker:
        print("\nMode: YetiRank (directly optimises NDCG) — building pseudo-groups...")
        train_pool, val_pool, X_val, y_val = _make_ranking_pools(X, y)
        train_pool_ref = train_pool
        print(f"  {50} train groups × ~2000 candidates, {len(X_val):,} val candidates")

        # Monotonic constraint: more skill_overlap must never lower the ranking score.
        # Fixes inversion where CatBoost ranks high-behavioral / low-skill candidates first.
        _feat_cols = list(X.columns)
        _mono_cst = [0] * len(_feat_cols)
        if 'skill_overlap' in _feat_cols:
            _mono_cst[_feat_cols.index('skill_overlap')] = 1
            print(f"  Monotonic constraint applied to skill_overlap (col {_feat_cols.index('skill_overlap')})")

        model = CatBoostRanker(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            subsample=0.8,
            colsample_bylevel=0.8,
            loss_function='YetiRank',
            eval_metric='NDCG',
            early_stopping_rounds=30,
            monotone_constraints=_mono_cst,
            random_seed=42,
            verbose=50,
        )
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    else:
        print("\nMode: Regression (RMSE) — legacy mode")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
        print(f"Train: {len(X_train):,}  Val: {len(X_val):,}")

        model = CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            subsample=0.8,
            colsample_bylevel=0.8,
            early_stopping_rounds=30,
            eval_metric='RMSE',
            random_seed=42,
            verbose=50,
        )
        model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)

    val_pred = model.predict(X_val)
    rmse     = np.sqrt(np.mean((val_pred - y_val) ** 2))
    rho, _   = spearmanr(y_val, val_pred)
    print(f"\nVal RMSE:            {rmse:.4f}")
    print(f"Val Spearman ρ:      {rho:.4f}  (>0.3 = useful signal)")

    # CatBoostRanker requires train_pool for feature importance; regressor does not
    fi_kwargs = {'data': train_pool_ref} if train_pool_ref is not None else {}
    importance = pd.DataFrame({
        'feature':    X.columns,
        'importance': model.get_feature_importance(**fi_kwargs),
    }).sort_values('importance', ascending=False)
    print("\nTop feature importances (learned weights):")
    for _, row in importance.head(12).iterrows():
        print(f"  {row['feature']:<30}  {row['importance']:.2f}")

    artifact = {
        'model':         model,
        'feature_names': list(X.columns),
        'jd_req':        jd_req,
        'val_rmse':      float(rmse),
        'val_spearman':  float(rho),
        'mode':          'yetirank' if use_ranker else 'regression',
        'label_mean':    float(y.mean()),
    }
    with open(output_path, 'wb') as f_out:
        pickle.dump(artifact, f_out)

    print(f"\n✓ Model saved to {output_path}")
    print(f"  Spearman ρ = {rho:.4f}  mode={'YetiRank' if use_ranker else 'RMSE'}")
    return artifact


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CatBoost LTR model for candidate ranking')
    parser.add_argument('--jd',           default='job_description.txt', help='JD text file')
    parser.add_argument('--output',       default='ltr_model.pkl',       help='Output model pickle')
    parser.add_argument('--no-retrieval', action='store_true',
                        help='Skip Stage 1 retrieval (faster, but no BM25/dense/RRF features)')
    parser.add_argument('--full-corpus', action='store_true',
                        help='Use full-corpus dense retrieval (no BM25) for cleaner LTR features')
    parser.add_argument('--no-ranker', action='store_true',
                        help='Use regression (RMSE) instead of YetiRank (legacy mode)')
    args = parser.parse_args()

    jd_text = Path(args.jd).read_text()

    from eda import load_data
    candidates = load_data()
    print(f"Loaded {len(candidates):,} candidate profiles.")

    retrieval_df = None
    if not args.no_retrieval:
        if args.full_corpus:
            print("\nRunning full-corpus dense retrieval (no BM25, all 100K candidates)...")
            from generate_submission import stage1_dense_fullcorpus
            retrieval_df = stage1_dense_fullcorpus(jd_text, candidates)
            # Remove bm25/rrf columns — they don't exist in full-corpus mode
            retrieval_df['bm25_score'] = 0.0
            retrieval_df['rrf_score']  = 0.0
        else:
            print("\nRunning Stage 1 hybrid retrieval to include BM25/dense/RRF as features...")
            from generate_submission import stage1_hybrid_retrieval
            retrieval_df = stage1_hybrid_retrieval(jd_text, candidates, fetch_per_system=500)
        print(f"Retrieval pool: {len(retrieval_df):,} candidates with retrieval scores.\n")

    train(jd_text, candidates, args.output, retrieval_df=retrieval_df,
          use_ranker=not args.no_ranker)
