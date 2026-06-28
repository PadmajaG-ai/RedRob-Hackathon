"""
CI fairness gate.

Asserts the pipeline produces no statistically significant disparate impact
(4/5ths rule: DI >= 0.80) at top-100 of the retrieval pool across three
diverse holdout JDs covering different seniority levels and skill profiles.

The test runs the FULL retrieval stage (BM25 + dense RRF) on a small
N_SAMPLE-candidate subset so it stays fast on CPU (~30-60 s total for all
3 JDs).  Groups smaller than FAIRNESS_MIN_GROUP_SIZE are excluded from the
DI calculation — too few observations for meaningful statistics.

Usage:
    pytest tests/test_fairness_gate.py -v
    pytest tests/test_fairness_gate.py -v -s   # shows audit table on failure
"""

import sys
import pytest
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Config ──────────────────────────────────────────────────────────────────
N_SAMPLE = 1_000    # candidates to sample; keeps retrieval fast on CPU
SEED     = 42

JD_DIR   = Path(__file__).parent / "fixtures" / "holdout_jds"
JD_FILES = sorted(JD_DIR.glob("*.txt"))


# ── Shared fixtures (loaded once per session) ───────────────────────────────

@pytest.fixture(scope="session")
def all_cands() -> pd.DataFrame:
    from eda import load_data
    return load_data()


@pytest.fixture(scope="session")
def sample_cands(all_cands: pd.DataFrame) -> pd.DataFrame:
    """Reproducible subset used for retrieval — fast on CPU."""
    n = min(N_SAMPLE, len(all_cands))
    return all_cands.sample(n=n, random_state=SEED).reset_index(drop=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _pool_ranking(jd_text: str, cands: pd.DataFrame) -> pd.DataFrame:
    """Stage 1 retrieval -> [candidate_id, rank] sorted by RRF score."""
    from generate_submission import stage1_hybrid_retrieval
    retrieved = stage1_hybrid_retrieval(jd_text, cands, fetch_per_system=100)
    ranked = retrieved.reset_index(drop=True).copy()
    ranked["rank"] = range(1, len(ranked) + 1)
    return ranked[["candidate_id", "rank"]]


# ── Tests ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("jd_path", JD_FILES, ids=[p.stem for p in JD_FILES])
def test_retrieval_pool_fairness(
    jd_path: Path,
    sample_cands: pd.DataFrame,
    all_cands: pd.DataFrame,
) -> None:
    """
    ASSERTION: Every audited attribute must satisfy DI >= 0.80 (4/5ths rule)
    at top-100 of the retrieval pool.

    Failure means Stage 1 retrieval itself is disproportionately filtering out
    a demographic group before ranking even begins.
    """
    from fairness_audit import (
        run_audit, print_report, check_violations,
        FAIRNESS_DI_THRESHOLD, FAIRNESS_MIN_GROUP_SIZE,
    )

    jd_text  = jd_path.read_text()
    ranking  = _pool_ranking(jd_text, sample_cands)
    pool_n   = len(ranking)

    # Use the k values that fit within the actual pool size.
    k_vals   = tuple(k for k in (10, 50, 100) if k <= pool_n) or (pool_n,)
    report   = run_audit(all_cands, ranking, k_values=k_vals)

    k_check  = min(100, pool_n)
    violations = check_violations(report, k=k_check, threshold=FAIRNESS_DI_THRESHOLD)

    if violations:
        # Print full audit table so the failure message tells you exactly what
        # group is over- or under-represented.
        print(f"\n[FAIL] {jd_path.name} — fairness violations at top-{k_check}:")
        print_report(report, focus_k=k_check)

    assert not violations, (
        f"Disparate impact violation(s) in retrieval pool for '{jd_path.name}' "
        f"at top-{k_check} (threshold={FAIRNESS_DI_THRESHOLD:.2f}, "
        f"min group size={FAIRNESS_MIN_GROUP_SIZE}):\n"
        + "\n".join(
            f"  {v['attribute']}: DI={v['disparate_impact']:.3f}  "
            f"— {v['description']}"
            for v in violations
        )
    )
