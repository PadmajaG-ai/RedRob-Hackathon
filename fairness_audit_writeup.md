# Fairness audit layer — design and implementation

## Why this matters

For a hiring system, the question every serious evaluator asks is:

> Does the ranking systematically advantage or disadvantage candidates based on attributes that have nothing to do with job competence?

Most hackathon submissions never answer it. A fairness audit is a quick win with outsized impact — it signals responsible-AI maturity, and it runs almost entirely on data you already have. This document describes the audit layer, the metrics it reports, the results on the current submission, and how to wire it into the pipeline.

---

## What it measures

The audit computes four complementary views, each at several cut-offs (top-10, top-50, top-100):

| Metric | What it answers | Threshold |
|---|---|---|
| **Selection rate** | What share of each group reaches the top-K? | — |
| **Disparate impact ratio** | min / max selection rate across groups (the "4/5ths rule") | < 0.80 is a legal red flag |
| **Statistical parity gap** | Absolute spread between the highest and lowest selection rate | Lower is better |
| **Exposure parity** | Rank-aware: is recruiter *attention* shared in proportion to group size? | Gap near 0 is fair |

### Why exposure parity is the differentiator

Selection rate treats rank 1 and rank 10 as equally "selected." But recruiters look at the top of a shortlist far more than the bottom — being ranked first gets roughly 3× the attention of being ranked tenth. Exposure parity weights each position by an NDCG-style logarithmic discount (`1 / log2(rank + 1)`) and checks whether each group's total exposure matches its share of the pool.

This catches a failure mode that pure selection-rate fairness misses entirely: a system can place equal numbers of each group in the top-50 yet still bury one group at the bottom of every shortlist. Demonstrating that you understand ranking fairness is not the same as classification fairness is exactly the kind of nuance that stands out.

---

## Results on the current submission

Running the audit on `submission_v3.csv` (100 ranked candidates) surfaced two findings:

**Experience band — flagged.** Senior candidates (6+ years) take all 10 top spots while mid-level candidates (3–6 years) get zero, failing the 4/5ths rule. This may be *legitimate* — the job description requires 5+ years of experience, so the disparity could reflect a genuine job requirement rather than bias. Surfacing this distinction is the point: a fairness layer flags the pattern so a human can judge whether it's justified, rather than hiding it.

**Location and company — inconclusive in the demo run.** These showed 0 in the top-10 because the top-15 candidates use LLM-generated reasoning that omits location and company fields, so the demo text-parser couldn't extract them. In production these come from structured candidate fields, not parsed text, and every candidate has them.

---

## Three deliberate design choices

1. **It refuses to infer gender, caste, or religion from names.** Name-based inference is unreliable and itself discriminatory. The module audits these dimensions only when a candidate has self-reported the attribute, and even then keeps results aggregate-only. This is documented prominently in the module header.

2. **It is attribute-agnostic.** Adding a new protected dimension is a single `GroupSpec` entry — a name, a function that maps a candidate row to a group label, and a description. Nothing is hard-coded to a specific attribute.

3. **It is rank-aware, not just selection-aware.** The exposure metric reflects that ranking fairness is a genuinely harder problem than classification fairness.

---

## How the code is organized

| Section | Responsibility |
|---|---|
| `GroupSpec` + derivers | Turn raw columns into categorical group labels (location tier, company type, experience band ship by default) |
| `audit_attribute()` | Compute all four metrics for one attribute at one K |
| `run_audit()` | Merge candidate profiles with the ranking, derive every group, audit every attribute at every K |
| `print_report()` | Human-readable console output with auto-flagging |
| `extract_attributes_from_reasoning()` | Demo-only fallback that parses attributes from reasoning text — replace with structured fields in production |

### Expected inputs

A candidate profile dataframe keyed on `candidate_id` with any attribute columns you want to audit (`location_city`, `company`, `experience_years`, `college_tier`, …), plus the ranking output (`candidate_id`, `rank`). You choose which columns become protected groups via the `GroupSpec` list.

---

## Wiring it into the pipeline

1. At the end of `generate_submission.py`, after the final ranking is produced, call:

   ```python
   from fairness_audit import run_audit, print_report
   report = run_audit(candidates_df, ranking_df)
   print_report(report, focus_k=10)
   ```

2. Run the audit against the full ~1000-candidate retrieval pool, not just the 100 ranked — comparing who was *selected* against who was *available* is what makes the selection-rate and exposure metrics meaningful.

3. Serialize `report` to JSON and turn the flagged attributes into a one-page panel for the deck. A single slide showing the 4/5ths ratio per attribute, plus the exposure-parity gap, communicates responsible-AI thinking that almost no competing submission will have.

---

## Recommended next step

Add an HTML renderer so the audit produces a visual dashboard (disparate-impact bars per attribute, exposure vs pool-share comparison) that can be screenshotted directly into the presentation — turning the audit from a backend check into a visible feature.
