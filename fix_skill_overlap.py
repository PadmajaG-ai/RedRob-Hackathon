# Fix skill_overlap ranking inversion in CatBoost
# ================================================

"""
ISSUE: Top-10 avg 4.2/20 skill matches, Bottom-10 avg 10.9/20 matches.
CatBoost is under-weighting technical fit vs behavioral signals.

Three fixes, ordered by effort:
"""

import catboost as cb
import numpy as np
import pandas as pd

# ============================================================================
# METHOD 1: Scale skill_overlap before training (FASTEST, 10 mins)
# ============================================================================
# CatBoost learns weights implicitly. If skill_overlap is small in magnitude
# (e.g., 0-7 scale) vs behavioral signals (0-10 scale), it gets less attention.
# Solution: scale it up to match or exceed other features.

def method_1_scale_features(X_train, X_test, skill_col='skill_overlap'):
    """Normalize skill_overlap to 0-100 scale before training."""
    # Before: skill_overlap in 0-20 range
    # After: skill_overlap in 0-100 range (same as trajectory, engagement, etc)
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    # 5x scale boost
    X_train[skill_col] = X_train[skill_col] * 5
    X_test[skill_col] = X_test[skill_col] * 5
    
    return X_train, X_test

# Usage:
# X_train_scaled, X_test_scaled = method_1_scale_features(X_train, X_test)
# model = cb.CatBoostRegressor(...)
# model.fit(X_train_scaled, y_train)


# ============================================================================
# METHOD 2: Add multiplicative penalty for low skill match (MODERATE, 1 hour)
# ============================================================================
# Don't just scale — make skill_overlap *condition* the final score.
# Low skill match should suppress the score, not just contribute less.

def method_2_skill_floor_penalty(predictions, X_test, skill_col='skill_overlap',
                                  threshold=5, penalty=0.7):
    """
    If skill_overlap < threshold, penalize the prediction.
    E.g., if a candidate matches < 5 skills, multiply their score by 0.7.
    """
    X_test = X_test.copy()
    penalized = predictions.copy()
    
    low_skill_mask = X_test[skill_col] < threshold
    penalized[low_skill_mask] = penalized[low_skill_mask] * penalty
    
    return penalized

# Usage:
# y_pred = model.predict(X_test)
# y_pred_penalized = method_2_skill_floor_penalty(y_pred, X_test, threshold=5, penalty=0.7)
# This is a POST-processing step on the final scores.


# ============================================================================
# METHOD 3: Hard filter before ranking (MOST DIRECT, 30 mins)
# ============================================================================
# Don't let candidates below a skill threshold into the top-K.
# This forces CatBoost to rank only qualified candidates.

def method_3_hard_skill_filter(X_ranked, skill_col='skill_overlap', 
                               top_k=100, min_skills_top_20=5):
    """
    1. Let CatBoost rank all 1000 candidates.
    2. Within top-20: remove anyone with skill_overlap < min_skills_top_20.
    3. Fill the gap from the next-ranked qualified candidates.
    4. Allow relaxed thresholds for ranks 21-100.
    """
    X_ranked = X_ranked.copy()
    X_ranked_sorted = X_ranked.sort_values('catboost_score', ascending=False).reset_index(drop=True)
    
    # Top-20 enforcement
    top_20 = X_ranked_sorted.iloc[:20].copy()
    top_20_qualified = top_20[top_20[skill_col] >= min_skills_top_20]
    top_20_unqualified = top_20[top_20[skill_col] < min_skills_top_20]
    
    # Backfill from rank 21+ with qualified candidates
    remaining = X_ranked_sorted.iloc[20:].copy()
    backfill = remaining[remaining[skill_col] >= min_skills_top_20].head(len(top_20_unqualified))
    
    # Reorder: top-20 qualified, then backfill, then rest of 21-100
    rest = remaining.drop(backfill.index)
    
    final_ranked = pd.concat([
        top_20_qualified,
        backfill,
        rest.head(top_k - len(top_20_qualified) - len(backfill))
    ], ignore_index=True)
    
    return final_ranked.head(top_k)

# Usage:
# X_test['catboost_score'] = model.predict(X_test)
# X_test_reranked = method_3_hard_skill_filter(X_test, min_skills_top_20=5, top_k=100)


# ============================================================================
# METHOD 4: Retrain with feature constraints (MOST ROBUST, 2 hours)
# ============================================================================
# CatBoost supports monotonic constraints and feature penalties.
# Use this to force the model to respect skill_overlap.

def method_4_constrained_catboost(X_train, y_train, X_test, 
                                   skill_col_idx=None, skill_col_name='skill_overlap'):
    """
    Train CatBoost with:
    - Monotonic constraint: skill_overlap must increase score (if this makes sense)
    - Feature scaling/boosting
    - Custom metric that penalizes ranking low-skill candidates high
    """
    # Find column index
    if skill_col_idx is None:
        skill_col_idx = X_train.columns.get_loc(skill_col_name)
    
    # Monotonic constraints: 1 = must increase score, -1 = must decrease, 0 = no constraint
    # For skill_overlap, we want MORE skills to mean HIGHER score: monotonic_cst = 1
    monotonic_constraints = [0] * len(X_train.columns)
    monotonic_constraints[skill_col_idx] = 1  # skill_overlap must be monotonic increasing
    
    # Feature scales: boost skill_overlap weight by 2x
    feature_weights = [1.0] * len(X_train.columns)
    feature_weights[skill_col_idx] = 2.0
    
    model = cb.CatBoostRegressor(
        iterations=100,
        learning_rate=0.1,
        depth=6,
        monotone_constraints=monotonic_constraints,  # Requires specific CatBoost versions
        # feature_weights=feature_weights,  # Not all versions support this
        random_state=42,
        verbose=10
    )
    
    model.fit(X_train, y_train, eval_set=(X_test, y_test))
    return model

# Note: monotone_constraints requires CatBoost >= 0.19
# If not available, use the scaling/penalty methods above instead.


# ============================================================================
# RECOMMENDATION: Combine METHOD 1 + METHOD 3
# ============================================================================
# Fastest path to fix:
# 1. Scale skill_overlap by 5x (method 1) — 10 mins, retrain, test NDCG@10
# 2. If still inverted, add hard filter (method 3) — 30 mins, top-20 must have ≥ 5 skills
# 3. Measure impact on NDCG@10 and Spearman ρ

def combined_fix(X_train, y_train, X_test, y_test, 
                 skill_col='skill_overlap', top_k=100):
    """
    End-to-end fix: scale, retrain, filter, validate.
    """
    # Step 1: Scale skill_overlap
    X_train_scaled, X_test_scaled = method_1_scale_features(X_train, X_test, skill_col)
    
    # Step 2: Retrain CatBoost
    model = cb.CatBoostRegressor(iterations=100, verbose=0)
    model.fit(X_train_scaled, y_train, eval_set=(X_test_scaled, y_test))
    
    # Step 3: Predict and apply hard filter
    X_test_ranked = X_test_scaled.copy()
    X_test_ranked['catboost_score'] = model.predict(X_test_scaled)
    
    # Hard filter: top-20 must have >= 5 skills
    final_ranking = method_3_hard_skill_filter(X_test_ranked, skill_col=skill_col, 
                                               top_k=top_k, min_skills_top_20=5)
    
    # Step 4: Validate
    from sklearn.metrics import ndcg_score
    y_true_top10 = y_test.iloc[final_ranking.head(10).index]
    y_pred_top10 = final_ranking.head(10)['catboost_score'].values
    
    ndcg = ndcg_score([y_true_top10], [y_pred_top10])
    print(f"NDCG@10 after fix: {ndcg:.4f}")
    print(f"Avg skill matches in top-10: {final_ranking.head(10)[skill_col].mean():.1f}/20")
    print(f"Avg skill matches in bottom-10: {final_ranking.tail(10)[skill_col].mean():.1f}/20")
    
    return model, final_ranking


# ============================================================================
# IMMEDIATE ACTION CHECKLIST
# ============================================================================
"""
1. [ ] Run current model, note top-10 and bottom-10 avg skill_overlap
       y_pred = model.predict(X_test)
       X_test['pred_score'] = y_pred
       top_10_skills = X_test.nlargest(10, 'pred_score')['skill_overlap'].mean()
       bot_10_skills = X_test.nsmallest(10, 'pred_score')['skill_overlap'].mean()
       print(f"Top-10: {top_10_skills:.1f}, Bottom-10: {bot_10_skills:.1f}")

2. [ ] Apply METHOD 1 (scaling): 5x skill_overlap, retrain, re-check

3. [ ] If inversion persists, apply METHOD 3 (hard filter): top-20 >= 5 skills

4. [ ] Re-calculate NDCG@10 on validation set

5. [ ] Update presentation: "Added skill match floor for top-20 to ensure 
       technical baseline qualification"
"""
