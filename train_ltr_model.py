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

Eval integrity (--test-split):
  A strict ID-based holdout is carved out BEFORE any feature extraction.
  Test candidates are never seen during training — their labels are used only
  for the final Spearman ρ report. This prevents the memorisation issue where
  the eval shortlist overlaps with the training pool.

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


def compute_label(candidate: dict) -> float:
    """
    End-of-funnel behavioral label.

    interview_completion_rate: did the candidate complete interviews on this platform?
    offer_acceptance_rate:     did they accept offers when given?

    Together these represent platform-observed candidate quality — independent of
    any JD-specific features, so they can serve as a proxy for "is this a serious,
    hireable candidate" without leaking JD-specific signals into the label.
    """
    def _safe(val):
        try:
            v = float(val or 0)
        except (TypeError, ValueError):
            v = 0.0
        return max(0.0, min(1.0, v))

    interview_rate = _safe(candidate.get('interview_completion_rate'))
    offer_rate     = _safe(candidate.get('offer_acceptance_rate'))
    return interview_rate * 0.6 + offer_rate * 0.4


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

        label = compute_label(cand)
        feature_rows.append(feats)
        label_rows.append(label)

        if (i + 1) % 20_000 == 0:
            print(f"  {i+1:,}/{len(candidates_df):,} done...")

    X = pd.DataFrame(feature_rows)
    y = np.array(label_rows, dtype=np.float32)
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
    test_split: float = 0.20,
    seed: int = 42,
) -> dict:
    """
    Train CatBoost model for candidate ranking.

    test_split: fraction held out as strict test set (never seen during training).
                IDs are split BEFORE feature extraction to prevent any leakage.
    use_ranker: True  → CatBoostRanker with YetiRank (optimises NDCG directly)
                False → CatBoostRegressor with RMSE (legacy, faster)
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

    # ── Strict ID-based split BEFORE any feature extraction ──────────────────
    n_total = len(candidates_df)
    n_test  = max(500, int(n_total * test_split))
    n_train = n_total - n_test

    rng          = np.random.default_rng(seed)
    shuffled_pos = rng.permutation(n_total)
    train_pos    = set(shuffled_pos[:n_train].tolist())

    train_df = candidates_df.iloc[list(train_pos)].reset_index(drop=True)
    test_df  = candidates_df.drop(index=list(train_pos)).reset_index(drop=True)

    id_col = 'candidate_id' if 'candidate_id' in candidates_df.columns else None
    test_ids = list(test_df[id_col]) if id_col else []

    print(f"\n{'='*60}")
    print(f"STRICT TRAIN/TEST SPLIT  (seed={seed})")
    print(f"  Train : {len(train_df):,} candidates  ({100*(1-test_split):.0f}%)")
    print(f"  Test  : {len(test_df):,} candidates  ({100*test_split:.0f}%) — NEVER seen during training")
    print(f"{'='*60}\n")

    # Restrict retrieval features to train candidates so test labels can't leak
    train_retrieval = None
    if retrieval_df is not None and id_col:
        train_ids_set   = set(train_df[id_col].tolist())
        train_retrieval = retrieval_df[retrieval_df[id_col].isin(train_ids_set)].reset_index(drop=True)
        print(f"Retrieval pool restricted to {len(train_retrieval):,} train candidates "
              f"(full pool was {len(retrieval_df):,}).")
    elif retrieval_df is not None:
        train_retrieval = retrieval_df

    # ── Feature extraction (train only) ──────────────────────────────────────
    X_train, y_train = build_dataset(train_df, jd_req, retrieval_df=train_retrieval)

    print(f"\nTrain dataset : {X_train.shape[0]:,} candidates × {X_train.shape[1]} features")
    print(f"Label stats   — mean: {y_train.mean():.4f}  std: {y_train.std():.4f}  "
          f"min: {y_train.min():.4f}  max: {y_train.max():.4f}")
    print(f"Candidates with label > 0.5: {(y_train > 0.5).sum():,} "
          f"({100*(y_train > 0.5).mean():.1f}%)")

    train_pool_ref = None

    if use_ranker:
        print("\nMode: YetiRank (directly optimises NDCG) — building pseudo-groups...")
        train_pool, val_pool, X_val, y_val = _make_ranking_pools(X_train, y_train, seed=seed)
        train_pool_ref = train_pool
        print(f"  50 train groups × ~2000 candidates, {len(X_val):,} internal val candidates")

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
            random_seed=seed,
            verbose=50,
        )
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    else:
        print("\nMode: Regression (RMSE) — legacy mode")
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=seed)
        print(f"Internal split — Train: {len(X_tr):,}  Val: {len(X_val):,}")

        model = CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            subsample=0.8,
            colsample_bylevel=0.8,
            early_stopping_rounds=30,
            eval_metric='RMSE',
            random_seed=seed,
            verbose=50,
        )
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)

    # Internal validation metrics
    val_pred = model.predict(X_val)
    val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
    val_rho, _ = spearmanr(y_val, val_pred)
    print(f"\nInternal val RMSE       : {val_rmse:.4f}")
    print(f"Internal val Spearman ρ : {val_rho:.4f}  (within training distribution — optimistic)")

    # ── Strict held-out test evaluation ──────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"STRICT HELD-OUT TEST  ({len(test_df):,} never-seen candidates)")
    print(f"{'='*60}")
    # At inference time all candidates get retrieval features if available,
    # so pass the full retrieval_df here to simulate real inference conditions.
    X_test, y_test = build_dataset(test_df, jd_req, retrieval_df=retrieval_df)
    test_pred = model.predict(X_test)
    test_rmse = np.sqrt(np.mean((test_pred - y_test) ** 2))
    test_rho, _ = spearmanr(y_test, test_pred)
    print(f"Held-out test RMSE      : {test_rmse:.4f}")
    print(f"Held-out test Spearman ρ: {test_rho:.4f}  <-- honest unbiased benchmark (report this)")

    delta = test_rho - val_rho
    if delta < -0.10:
        print(f"\nWARNING: large val→test gap ({delta:+.4f}) — possible overfitting or distribution shift.")
    else:
        print(f"Train→test gap          : {delta:+.4f}  (within acceptable range)")

    # Feature importance
    fi_kwargs = {'data': train_pool_ref} if train_pool_ref is not None else {}
    importance = pd.DataFrame({
        'feature':    X_train.columns,
        'importance': model.get_feature_importance(**fi_kwargs),
    }).sort_values('importance', ascending=False)
    print("\nTop feature importances (learned weights):")
    for _, row in importance.head(12).iterrows():
        print(f"  {row['feature']:<30}  {row['importance']:.2f}")

    artifact = {
        'model':              model,
        'feature_names':      list(X_train.columns),
        'jd_req':             jd_req,
        'val_rmse':           float(val_rmse),
        'val_spearman':       float(val_rho),
        'test_rmse':          float(test_rmse),
        'test_spearman':      float(test_rho),
        'test_split':         test_split,
        'test_candidate_ids': test_ids,
        'mode':               'yetirank' if use_ranker else 'regression',
    }
    with open(output_path, 'wb') as f_out:
        pickle.dump(artifact, f_out)

    print(f"\n✓ Model saved to {output_path}")
    print(f"  Internal val Spearman ρ  = {val_rho:.4f}")
    print(f"  Held-out test Spearman ρ = {test_rho:.4f}  <-- report this")
    print(f"  Mode: {'YetiRank' if use_ranker else 'RMSE'}")
    return artifact


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CatBoost LTR model for candidate ranking')
    parser.add_argument('--jd',           default='job_description.txt', help='JD text file')
    parser.add_argument('--output',       default='ltr_model_retrained.pkl', help='Output model pickle')
    parser.add_argument('--no-retrieval', action='store_true',
                        help='Skip Stage 1 retrieval (faster, but no BM25/dense/RRF features)')
    parser.add_argument('--full-corpus',  action='store_true',
                        help='Use full-corpus dense retrieval (no BM25) for cleaner LTR features')
    parser.add_argument('--no-ranker',    action='store_true',
                        help='Use regression (RMSE) instead of YetiRank (legacy mode)')
    parser.add_argument('--test-split',   type=float, default=0.20,
                        help='Fraction of candidates held out as strict test set (default 0.20)')
    parser.add_argument('--seed',         type=int,   default=42,
                        help='Random seed for reproducible splits (default 42)')
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
            retrieval_df['bm25_score'] = 0.0
            retrieval_df['rrf_score']  = 0.0
        else:
            print("\nRunning Stage 1 hybrid retrieval to include BM25/dense/RRF as features...")
            from generate_submission import stage1_hybrid_retrieval
            retrieval_df = stage1_hybrid_retrieval(jd_text, candidates, fetch_per_system=500)
        print(f"Retrieval pool: {len(retrieval_df):,} candidates with retrieval scores.\n")

    train(
        jd_text, candidates, args.output,
        retrieval_df=retrieval_df,
        use_ranker=not args.no_ranker,
        test_split=args.test_split,
        seed=args.seed,
    )
