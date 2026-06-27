# Architecture improvement recommendations — File 2 ranking system

## Overview

Analysis of the multi-stage candidate ranking pipeline (hybrid retrieval → CatBoost LTR → LLM reasoning) based on the submission CSV and system architecture. Issues are grouped by component and ordered by priority within each section.

---

## Scoring layer

### 🔴 Critical — Technical skill match is effectively meaningless

**Problem:** Max observed score is 4.3/10, mean is 2.3/10, and 77% of candidates score below 3.0. The dimension's correlation with the final score is only 0.28 — the lowest of any factor. In practice it contributes almost no rank separation.

**Fix:** Re-calibrate the skill match scorer. Use partial credit across required skill *categories*, not just raw count/20. A candidate matching 5 of 7 core ML skills should score ~7, not 3.5. Consider semantic cluster matching — "FAISS" should count toward a "vector DB" category even if FAISS isn't explicitly listed in the JD. The ceiling of 4.3/10 suggests the feature extractor has a bug or an overly strict matching rule.

**Impact:** High

---

### 🔴 Critical — Career trajectory is a de facto gating criterion

**Problem:** Every top-10 candidate has `career_trajectory = 10.0`. The dimension correlates 0.69 with final score — far above any other factor. Candidates with trajectory below 8.0 almost never rank above position 50, regardless of how strong their other signals are. The other five dimensions are cosmetic for top rankings.

**Fix:** Reduce trajectory's effective weight or apply diminishing returns above ~8.0. Consider a multiplicative floor instead of a linear contribution — trajectory should disqualify weak candidates, not crown strong ones. Also audit what inputs produce 10.0 so reliably across 18 candidates; it suggests the scoring function is hitting a ceiling rather than discriminating meaningfully.

**Impact:** High

---

### 🟡 Important — Behavioral engagement negatively correlates with final score

**Problem:** Correlation = −0.13. High-engagement candidates tend to rank *lower*. A dimension that anti-correlates with the target should not be in the composite formula until the cause is understood.

**Fix:** Audit what "behavioral engagement" captures in the raw data. If it reflects platform activity (opens, saves, searches), it may be picking up active job-seekers rather than high-value passive candidates. Consider splitting into two signals: *activity volume* (noisy) and *recruiter response rate* (more predictive). If the negative correlation persists after investigation, drop or cap this dimension.

**Impact:** Medium

---

### 🟡 Important — Score distribution is severely compressed at the bottom

**Problem:** 79 of 100 candidates score below 0.5; 25 score below 0.1. There is no meaningful signal to distinguish rank 30 from rank 80. The bottom 80% of the list is unusable.

**Fix:** Apply score normalization or percentile calibration before producing the final ranking. Min-max scaling or a rank-to-score mapping (e.g. sigmoid over raw CatBoost scores) would spread the distribution. The goal is that adjacent ranks carry meaningfully different scores across the full list, not just the top 20.

**Impact:** Medium

---

## Retrieval layer

### 🟡 Important — RRF fusion may be discarding strong dense-only signals

**Problem:** Reciprocal Rank Fusion treats all retrievers equally in rank space. But bge-m3 dense retrieval surfaces semantically strong candidates that BM25 ranks poorly, and RRF dilutes those via rank averaging. The standalone dense baseline achieves NDCG@10 = 0.535 — potentially because its best hits get penalised by RRF when BM25 disagrees.

**Fix:** Test score-based fusion (weighted combination of normalized BM25 and dense scores) alongside RRF and compare on the held-out eval set. Consider adaptive weighting: for JDs with many exact-match required skills, weight BM25 higher; for senior/lead roles with fuzzy skill sets, weight dense higher.

**Impact:** Medium

---

### 🔵 Nice to fix — No feedback loop from LTR model back to retrieval

**Problem:** Stage 1 retrieval and Stage 2 LTR are fully decoupled. The ~1000-candidate pool from Stage 1 is fixed — if a strong candidate ranked 1001st in retrieval, the LTR model never sees them. This is a hard recall ceiling.

**Fix:** Add a diversity-aware expansion pass after RRF: sample additional candidates from underrepresented clusters (by domain, seniority, company type) before passing to LTR. Alternatively, fine-tune bge-m3 embeddings on labeled recruiting pairs — bge-reranker-v2-m3 is already in the stack (`finetune_reranker.py`).

**Impact:** Low

---

## LLM layer

### 🟡 Important — LLM reasoning only covers top-15, not top-100

**Problem:** The submission CSV has 100 ranked candidates but Qwen2.5 reasoning is generated only for the top 15. Candidates ranked 16–100 receive a formulaic dimension prefix as their "reasoning", providing no real explainability and making the output format inconsistent.

**Fix:** Either extend LLM reasoning to all 100 via batch processing, or generate a templated natural-language summary from the 6 dimension scores for ranks 16–100 (no LLM cost, consistent format). At minimum, be explicit in the output that ranks 16–100 use dimension scores only — not LLM-generated reasoning.

**Impact:** Medium

---

### 🔵 Nice to fix — Pass 1 (ranking confirmation) is architecturally redundant

**Problem:** The pipeline runs an LLM confirmation pass before generating reasoning. But ranking is already handled deterministically by CatBoost. An LLM confirmation step adds latency, risks hallucinated re-rankings that contradict the model, and conflates two separate responsibilities.

**Fix:** Remove Pass 1 entirely. Feed the CatBoost-ranked order directly to Pass 2 (reasoning generation). If LLM-based ranking is worth testing, treat it as a separate ablation experiment — not something baked silently into the production pipeline.

**Impact:** Low

---

## Evaluation

### 🔴 Critical — Behavioral labels used for training may leak into evaluation

**Problem:** CatBoost is trained on `interview_completion_rate × offer_acceptance_rate` across all 100K candidates. If the evaluation pool (the ~1000-candidate shortlist) overlaps with training data, the reported NDCG@10 = 0.87 is partly measuring memorization, not generalization.

**Fix:** Hold out a strict temporal or ID-based test split *before* any training. Evaluation must use candidates the model has never seen during training. Report NDCG on this held-out set separately from in-distribution results. This is the single most important step before claiming the benchmark numbers are valid.

**Impact:** High

---

### 🟡 Important — Spearman ρ = 0.32 is weaker than it appears

**Problem:** The deck presents ρ = 0.32 as a positive result, but this means only ~10% of ranking variance is explained by behavioral labels. For a system claiming to surface the best candidates, this is a modest correlation — especially if behavioral labels (interview completion + offer acceptance) are themselves noisy proxies for actual job performance.

**Fix:** Collect or proxy richer outcome labels — hiring decision, 90-day performance, re-application rate. Even a small labeled sample (50–100 outcomes) used for calibration would significantly improve correlation quality. Report confidence intervals around ρ, not just the point estimate.

**Impact:** Medium

---

### 🔵 Nice to fix — Trap detection metric is binary and fragile

**Problem:** "Zero trap candidates in top-10" is a hard-coded rule filter result, not a learned property. It measures rule precision over a very small window, not the model's ability to distinguish genuine AI practitioners from keyword stuffers at scale.

**Fix:** Add a soft trap score as a CatBoost feature rather than a post-hoc penalty. Report Trap@K for K = 5, 10, 20, 50 so the metric isn't artificially easy. Consider precision-recall curves for trap detection as a standalone diagnostic separate from the main ranking evaluation.

**Impact:** Low

---

## Recommended priority order

| Priority | Area | Action |
|---|---|---|
| 1 | Scoring | Fix skill match calibration (re-score with category-based partial credit) |
| 2 | Evaluation | Enforce strict train/eval split; re-run benchmarks |
| 3 | Scoring | Audit and reduce career trajectory dominance |
| 4 | Scoring | Investigate and resolve behavioral engagement anti-correlation |
| 5 | Scoring | Apply score normalization across the full top-100 |
| 6 | LLM | Extend reasoning (or templated summaries) to all 100 candidates |
| 7 | Retrieval | Test score-based fusion as an alternative to RRF |
| 8 | Evaluation | Report Spearman ρ with confidence intervals; collect richer outcome labels |
| 9 | LLM | Remove LLM Pass 1 (ranking confirmation) |
| 10 | Retrieval | Add diversity-aware expansion before LTR |
| 11 | Evaluation | Replace binary trap metric with Trap@K curve |
