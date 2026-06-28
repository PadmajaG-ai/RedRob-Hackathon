"""
fairness_audit.py
=================
A bias and fairness audit layer for the candidate ranking pipeline.

It answers one question judges (and any real recruiting team) will ask:
    "Does the ranking systematically advantage or disadvantage candidates
     based on attributes that have nothing to do with job competence?"

It reports four complementary views, computed at several cut-offs (top-10,
top-50, top-100):

  1. Selection rate           - share of each group that reaches the top-K.
  2. Disparate impact ratio   - min/max selection rate across groups.
                                The "4/5ths rule": < 0.80 is a legal red flag.
  3. Statistical parity diff  - max - min selection rate (absolute gap).
  4. Exposure parity          - rank-aware. Being #1 gets far more recruiter
                                attention than #10, so we weight by a
                                position-discount (like NDCG) and check whether
                                exposure is shared in proportion to group size.

Exposure parity is the one most ranking systems miss: you can pass the
selection-rate test and still bury one group at the bottom of every shortlist.

------------------------------------------------------------------------------
EXPECTED INPUT (production path)
------------------------------------------------------------------------------
A candidates dataframe with, at minimum:
    candidate_id : str
plus the real profile columns from eda.load_data():
    location, current_company, years_of_experience, current_industry, ...

and the ranking output (your submission CSV or scored DataFrame) with:
    candidate_id, rank

You choose which attribute columns become "protected groups" via GROUP_SPECS.
Nothing is hard-coded to a specific attribute — add or remove freely.

------------------------------------------------------------------------------
A NOTE ON GENDER / CASTE / RELIGION PROXIES
------------------------------------------------------------------------------
Do NOT infer gender, caste, or religion from names. Name-based inference is
unreliable and itself discriminatory. Audit these dimensions only when the
candidate has *self-reported* the attribute for diversity purposes, and even
then keep it aggregate-only. This module ships with location, company type,
and experience band because those are defensible and present in your data.
"""

from __future__ import annotations
import json
import math
import dataclasses
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Config constants — change here, not scattered across call sites
# ---------------------------------------------------------------------------
FAIRNESS_DI_THRESHOLD   = 0.80   # 4/5ths rule: below this is a legal red flag
FAIRNESS_K_VALUES       = (10, 50, 100)
FAIRNESS_MIN_GROUP_SIZE = 10     # skip groups too small for meaningful statistics


# ---------------------------------------------------------------------------
# 1. Group definitions
# ---------------------------------------------------------------------------
# Each GroupSpec turns a raw column (or a derived value) into a categorical
# group label. `deriver` maps a candidate row -> group label (or None to skip).

@dataclass
class GroupSpec:
    name: str                          # e.g. "location_tier"
    deriver: Callable[[pd.Series], Optional[str]]
    description: str = ""


# --- Location tier (India-focused; extend the maps for other geographies) ---
_METRO = {
    "mumbai", "delhi", "new delhi", "bengaluru", "bangalore", "hyderabad",
    "chennai", "kolkata", "pune", "ahmedabad", "gurgaon", "gurugram", "noida",
}
_TIER2 = {
    "jaipur", "chandigarh", "indore", "kochi", "coimbatore", "vizag",
    "visakhapatnam", "nagpur", "lucknow", "bhopal", "vadodara", "surat",
    "thiruvananthapuram", "mysuru", "mysore", "mohali", "trivandrum",
    "bhubaneswar", "kochi",
}

def _city_tier(row: pd.Series) -> Optional[str]:
    # Real field: "location" (format "Gurgaon, Haryana" or "Toronto")
    raw = str(row.get("location", "")).strip()
    city = raw.split(",")[0].strip().lower()
    if not city or city == "nan":
        return None
    if city in _METRO:
        return "metro"
    if city in _TIER2:
        return "tier-2"
    return "tier-3+"


# --- Company type: product vs services vs startup ---------------------------
_SERVICES = {"tcs", "infosys", "wipro", "cognizant", "accenture", "capgemini",
             "hcl", "tech mahindra", "genpact", "mindtree", "ltimindtree"}
_PRODUCT = {"google", "meta", "microsoft", "amazon", "flipkart", "cred",
            "swiggy", "zomato", "ola", "byju's", "byjus", "phonepe", "razorpay",
            "yellow.ai", "inmobi", "vedantu", "unacademy", "paytm"}

def _company_type(row: pd.Series) -> Optional[str]:
    # Real field: "current_company"
    c = str(row.get("current_company", "")).strip().lower()
    if not c or c == "nan":
        return None
    if any(s in c for s in _SERVICES):
        return "services"
    if any(p in c for p in _PRODUCT):
        return "product"
    return "other"


# --- Experience band --------------------------------------------------------
def _experience_band(row: pd.Series) -> Optional[str]:
    # Real field: "years_of_experience" (float64)
    yrs = row.get("years_of_experience")
    try:
        yrs = float(yrs)
    except (TypeError, ValueError):
        return None
    if yrs < 3:
        return "junior (<3y)"
    if yrs < 6:
        return "mid (3-6y)"
    return "senior (6y+)"


DEFAULT_GROUP_SPECS = [
    GroupSpec("location_tier",  _city_tier,        "Metro / tier-2 / tier-3 city"),
    GroupSpec("company_type",   _company_type,     "Product / services / other"),
    GroupSpec("experience_band", _experience_band, "Career stage by years"),
]


# ---------------------------------------------------------------------------
# 2. Fairness metrics
# ---------------------------------------------------------------------------
@dataclass
class GroupResult:
    group: str
    pool_n: int
    selected_n: int
    selection_rate: float
    pool_share: float
    selected_share: float
    representation_ratio: float        # selected_share / pool_share
    exposure_share: float              # share of total position-discounted exposure

@dataclass
class AttributeAudit:
    attribute: str
    k: int
    groups: list[GroupResult]
    disparate_impact: float            # min/max selection rate (4/5ths rule)
    statistical_parity_diff: float     # max - min selection rate
    exposure_gap: float                # max - min (exposure_share - pool_share)
    flagged: bool = field(init=False)

    def __post_init__(self):
        self.flagged = self.disparate_impact < FAIRNESS_DI_THRESHOLD


def _position_discount(rank: int) -> float:
    """NDCG-style log discount: rank 1 -> 1/log2(2)=1.0, rank 10 -> ~0.30."""
    return 1.0 / math.log2(rank + 1)


def audit_attribute(df: pd.DataFrame, attribute: str, k: int) -> AttributeAudit:
    """
    df must have columns: [attribute, 'rank'] for the full ranked pool.
    Candidates with a null group label are dropped from that attribute's audit.
    Groups smaller than FAIRNESS_MIN_GROUP_SIZE are excluded from the DI
    calculation (too small to be statistically meaningful) but still shown.
    """
    sub = df[df[attribute].notna()].copy()
    pool_n = len(sub)
    selected = sub[sub["rank"] <= k]

    total_exposure = sub["rank"].map(_position_discount).sum()

    results: list[GroupResult] = []
    for g, gdf in sub.groupby(attribute):
        gpool = len(gdf)
        gsel = int((gdf["rank"] <= k).sum())
        gexp = gdf["rank"].map(_position_discount).sum()
        results.append(GroupResult(
            group=str(g),
            pool_n=gpool,
            selected_n=gsel,
            selection_rate=gsel / gpool if gpool else 0.0,
            pool_share=gpool / pool_n if pool_n else 0.0,
            selected_share=gsel / len(selected) if len(selected) else 0.0,
            representation_ratio=((gsel / len(selected)) / (gpool / pool_n))
                if len(selected) and gpool else 0.0,
            exposure_share=gexp / total_exposure if total_exposure else 0.0,
        ))

    # Only include groups with sufficient size in DI calculation
    eligible_rates = [r.selection_rate for r in results
                      if r.pool_n >= FAIRNESS_MIN_GROUP_SIZE]
    if len(eligible_rates) >= 2 and max(eligible_rates) > 0:
        di = min(eligible_rates) / max(eligible_rates)
    else:
        di = 1.0  # insufficient data — don't flag

    spd = (max(eligible_rates) - min(eligible_rates)) if len(eligible_rates) >= 2 else 0.0
    exp_gap = max((r.exposure_share - r.pool_share) for r in results) - \
              min((r.exposure_share - r.pool_share) for r in results) \
              if results else 0.0

    return AttributeAudit(attribute, k, results, di, spd, exp_gap)


# ---------------------------------------------------------------------------
# 3. Top-level runner
# ---------------------------------------------------------------------------
def run_audit(candidates: pd.DataFrame,
              ranking: pd.DataFrame,
              group_specs: list[GroupSpec] = DEFAULT_GROUP_SPECS,
              k_values: tuple[int, ...] = FAIRNESS_K_VALUES) -> dict:
    """
    candidates : full profile dataframe from eda.load_data()
                 (must have candidate_id, location, current_company,
                  years_of_experience, etc.)
    ranking    : DataFrame with at minimum [candidate_id, rank]
    returns a nested dict suitable for JSON export or rendering.
    """
    df = candidates.merge(ranking[["candidate_id", "rank"]],
                          on="candidate_id", how="inner")

    # Derive each protected group column.
    for spec in group_specs:
        df[spec.name] = df.apply(spec.deriver, axis=1)

    report: dict = {"n_ranked": len(df), "attributes": {}}
    for spec in group_specs:
        report["attributes"][spec.name] = {
            "description": spec.description,
            "by_k": {k: audit_attribute(df, spec.name, k) for k in k_values},
        }
    return report


# ---------------------------------------------------------------------------
# 4. Helpers for CI gate and logging
# ---------------------------------------------------------------------------
def check_violations(
    report: dict,
    k: int = 100,
    threshold: float = FAIRNESS_DI_THRESHOLD,
) -> list[dict]:
    """
    Return a list of violation dicts for all attributes where DI < threshold
    at the given k.  Empty list = all clear.

    Each dict: {attribute, k, disparate_impact, description}
    """
    violations = []
    for attr, payload in report["attributes"].items():
        by_k = payload["by_k"]
        if k not in by_k:
            continue
        audit: AttributeAudit = by_k[k]
        if audit.disparate_impact < threshold:
            violations.append({
                "attribute":        attr,
                "k":                k,
                "disparate_impact": audit.disparate_impact,
                "description":      payload["description"],
            })
    return violations


def _to_serializable(obj):
    """Recursively convert dataclasses / dicts / lists to JSON-safe types."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]
    return obj


def save_report(report: dict, path) -> None:
    """Write audit report as JSON to *path*."""
    with open(path, "w") as fh:
        json.dump(_to_serializable(report), fh, indent=2, default=str)


# ---------------------------------------------------------------------------
# 5. Pretty-printer
# ---------------------------------------------------------------------------
def print_report(report: dict, focus_k: int = 10) -> None:
    print(f"\nFAIRNESS AUDIT  -  {report['n_ranked']} ranked candidates\n" + "=" * 64)
    for attr, payload in report["attributes"].items():
        by_k = payload["by_k"]
        if focus_k not in by_k:
            continue
        audit: AttributeAudit = by_k[focus_k]
        flag = "  [FLAGGED: fails 4/5ths rule]" if audit.flagged else ""
        print(f"\n{attr}  ({payload['description']})  -  top-{focus_k}{flag}")
        print(f"  disparate impact ratio : {audit.disparate_impact:.2f}"
              f"   (>= {FAIRNESS_DI_THRESHOLD} ok)")
        print(f"  statistical parity gap : {audit.statistical_parity_diff:.2f}")
        print(f"  exposure gap           : {audit.exposure_gap:.2f}")
        print(f"  {'group':<14}{'pool':>6}{'in top':>8}{'sel.rate':>10}"
              f"{'repr.ratio':>12}{'exposure':>10}"
              f"{'note':>10}")
        for r in sorted(audit.groups, key=lambda x: -x.selection_rate):
            note = " (small)" if r.pool_n < FAIRNESS_MIN_GROUP_SIZE else ""
            print(f"  {r.group:<14}{r.pool_n:>6}{r.selected_n:>8}"
                  f"{r.selection_rate:>10.2f}{r.representation_ratio:>12.2f}"
                  f"{r.exposure_share:>10.2f}{note:>10}")


# ---------------------------------------------------------------------------
# 6. Fallback: extract attributes from reasoning text
#    (demo only — prefer structured candidate fields in production)
# ---------------------------------------------------------------------------
_LOC_RE = re.compile(r"Location match \(([^,]+),\s*([^)]+)\)|"
                     r"([A-Za-z]+),\s*([A-Za-z ]+)\)\s*[-—]\s*relocation")
_EXP_RE = re.compile(r"(\d+\.?\d*)\s*years?\s*\(([^)]+?)\s+at\s+([^)]+?)\)")

def extract_attributes_from_reasoning(ranking: pd.DataFrame) -> pd.DataFrame:
    """Demo-only fallback. Use eda.load_data() in production."""
    rows = []
    for _, r in ranking.iterrows():
        text = str(r.get("reasoning", ""))
        city = company = None
        exp = None
        m = _LOC_RE.search(text)
        if m:
            city = (m.group(1) or m.group(3) or "").strip()
        m = _EXP_RE.search(text)
        if m:
            exp = float(m.group(1))
            company = m.group(3).strip()
        rows.append({
            "candidate_id":        r["candidate_id"],
            "location":            city,
            "current_company":     company,
            "years_of_experience": exp,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7. CLI — uses real candidate data from eda.load_data()
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    submission_path = sys.argv[1] if len(sys.argv) > 1 else "submission_v2.csv"
    ranking = pd.read_csv(submission_path)

    from eda import load_data
    candidates = load_data()

    report = run_audit(candidates, ranking, k_values=FAIRNESS_K_VALUES)
    print_report(report, focus_k=10)
    print_report(report, focus_k=50)

    violations = check_violations(report, k=100)
    if violations:
        print(f"\n[WARN] {len(violations)} fairness concern(s) at top-100:")
        for v in violations:
            print(f"  {v['attribute']}: DI={v['disparate_impact']:.2f} "
                  f"(threshold {FAIRNESS_DI_THRESHOLD})")
    else:
        print(f"\n[OK] All attributes pass 4/5ths rule at top-100.")
