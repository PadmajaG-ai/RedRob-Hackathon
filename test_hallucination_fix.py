#!/usr/bin/env python3
"""
Test suite for the hallucination fix in llm_judge.py.

Layer 1 (no model needed): verify prompts are correctly grounded
Layer 2 (live, optional):  run a real LLM call and inspect the output

Run:
    python test_hallucination_fix.py            # Layer 1 only (fast)
    python test_hallucination_fix.py --live     # Layer 1 + Layer 2 (needs GPU/model)
"""

import re
import sys
import json
import argparse


# ── Fixtures ──────────────────────────────────────────────────────────────────

JD_TEXT = open("job_description.txt").read()

# Candidate whose current employer could be confused with the target company
CANDIDATE_AT_META = {
    "candidate_id": "CAND_TEST_001",
    "current_title": "Senior Applied Scientist",
    "current_company": "Meta",
    "years_of_experience": 8,
    "location": "Mumbai, India",
    "skills": ["pytorch", "llm", "embeddings", "rag", "fine-tuning", "lora",
               "faiss", "nlp", "reinforcement learning", "llamaIndex"],
    "summary": "Led ML ranking systems at Meta for 8 years. Built production embedding pipelines.",
    "current_industry": "technology",
    "recruiter_response_rate": 0.85,
    "github_activity_score": 72,
    "open_to_work_flag": True,
    "notice_period_days": 30,
    "profile_completeness_score": 90,
    "interview_completion_rate": 0.8,
    "offer_acceptance_rate": 0.7,
}

CANDIDATE_AT_GOOGLE = {
    "candidate_id": "CAND_TEST_002",
    "current_title": "Staff Software Engineer",
    "current_company": "Google",
    "years_of_experience": 6,
    "location": "Bangalore, India",
    "skills": ["python", "elasticsearch", "vector search", "faiss", "bge", "ndcg"],
    "summary": "Built search ranking systems at Google. Experience with hybrid retrieval.",
    "current_industry": "technology",
    "recruiter_response_rate": 0.6,
    "github_activity_score": 55,
    "open_to_work_flag": False,
    "notice_period_days": 60,
    "profile_completeness_score": 80,
    "interview_completion_rate": 0.7,
    "offer_acceptance_rate": 0.6,
}

CANDIDATES = [CANDIDATE_AT_META, CANDIDATE_AT_GOOGLE]


# ── Layer 1: Prompt inspection (no model) ─────────────────────────────────────

def test_prompts_contain_target_role():
    """REASONING_PROMPT and LISTWISE_RANK_PROMPT must have {role_title}/{hiring_company}."""
    from llm_judge import REASONING_PROMPT, LISTWISE_RANK_PROMPT

    for template_name, template in [
        ("REASONING_PROMPT", REASONING_PROMPT),
        ("LISTWISE_RANK_PROMPT", LISTWISE_RANK_PROMPT),
    ]:
        assert "{role_title}" in template, \
            f"{template_name} missing {{role_title}} — hallucination fix incomplete"
        assert "{hiring_company}" in template, \
            f"{template_name} missing {{hiring_company}} — hallucination fix incomplete"

    print("PASS  prompts contain {role_title} and {hiring_company}")


def test_reasoning_prompt_has_anti_hallucination_instruction():
    """REASONING_PROMPT must explicitly tell the LLM not to confuse candidate's employer."""
    from llm_judge import REASONING_PROMPT

    # Render with dummy values to check the instruction is present
    rendered = REASONING_PROMPT.format(
        role_title="Senior AI Engineer",
        hiring_company="Redrob AI",
        jd="dummy jd text",
        candidates="dummy candidate block",
    )
    assert "do not" in rendered.lower() or "not" in rendered.lower(), \
        "REASONING_PROMPT has no prohibition against echoing candidate's employer"
    # The rendered prompt must name the actual hiring company
    assert "Redrob AI" in rendered, \
        "Hiring company 'Redrob AI' not present in rendered prompt"
    print("PASS  REASONING_PROMPT has anti-hallucination instruction and names hiring company")


def test_candidate_summary_includes_skill_gap():
    """_candidate_summary must include matched/missing skills when required_skills provided."""
    from llm_judge import ListwiseLLMRanker

    required = ["embeddings", "rag", "faiss", "lora", "bge", "ndcg", "a/b testing"]
    summary = ListwiseLLMRanker._candidate_summary(1, CANDIDATE_AT_META, required_skills=required)

    assert "matched" in summary.lower() or "coverage" in summary.lower(), \
        "Summary does not include skill match count"
    assert "missing" in summary.lower(), \
        "Summary does not list missing skills"
    # Should mention at least one matched skill
    matched_in_summary = any(sk in summary for sk in ["embeddings", "rag", "faiss", "lora"])
    assert matched_in_summary, "Summary does not name any matched skills"

    print(f"PASS  _candidate_summary includes skill gap data")
    print(f"      Preview:\n{summary}")


def test_candidate_summary_without_required_skills_still_works():
    """_candidate_summary must not crash when required_skills is None."""
    from llm_judge import ListwiseLLMRanker

    summary = ListwiseLLMRanker._candidate_summary(1, CANDIDATE_AT_META)
    assert "Senior Applied Scientist" in summary
    print("PASS  _candidate_summary works without required_skills")


def test_jd_parsing_extracts_redrob():
    """parse_jd must extract the role title and company from the Redrob JD."""
    import re

    role_title = "the open role"
    hiring_company = "the hiring company"
    for line in JD_TEXT.splitlines()[:10]:
        # Only match the "Job Description: <title> —" header line
        m = re.match(r'Job Description[:\s]+(.+?)\s*[—–]', line)
        if m:
            role_title = m.group(1).strip()
        m2 = re.match(r'Company[:\s]+(.+)', line)
        if m2:
            hiring_company = m2.group(1).split('(')[0].strip()

    assert "redrob" in hiring_company.lower(), \
        f"Expected 'Redrob' in hiring_company, got: '{hiring_company}'"
    assert "engineer" in role_title.lower() or "ai" in role_title.lower(), \
        f"Role title looks wrong: '{role_title}'"

    print(f"PASS  JD parsed correctly → role='{role_title}', company='{hiring_company}'")


# ── Layer 2: Live LLM spot-check ──────────────────────────────────────────────

def test_live_reasoning_does_not_hallucinate(model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
    """
    Run the real LLM and check that:
      1. The output does NOT say "at Meta" or "at Google" as the TARGET role
      2. The output DOES reference "Redrob AI" or the role title
      3. The output contains a skill count or gap mention
    """
    from llm_judge import (
        ListwiseLLMRanker, REASONING_PROMPT, LISTWISE_SYSTEM, parse_jd
    )

    print(f"\n[live] Loading model: {model_name} ...")
    ranker = ListwiseLLMRanker(model_name=model_name)

    jd_req = parse_jd(JD_TEXT)
    required_skills = jd_req.get("skills", [])
    role_title = "Senior AI Engineer"
    hiring_company = "Redrob AI"

    cand_summaries = [
        ranker._candidate_summary(i + 1, c, required_skills)
        for i, c in enumerate(CANDIDATES)
    ]
    top_cand_block = "\n\n".join(cand_summaries)

    prompt = REASONING_PROMPT.format(
        jd=JD_TEXT[:700],
        candidates=top_cand_block,
        role_title=role_title,
        hiring_company=hiring_company,
    )

    print("[live] Sending prompt to LLM ...")
    reply = ranker._call_llm(
        [{"role": "system", "content": LISTWISE_SYSTEM},
         {"role": "user",   "content": prompt}],
        max_tokens=300,
    )
    print(f"\n[live] Raw LLM reply:\n{reply}\n")

    reply_lower = reply.lower()

    # Check 1: candidate's current employer not used as the TARGET role
    # (this was the original hallucination — "aligns with the need for Senior Applied Scientist at Meta")
    bad_patterns = [
        r"for\s+(the\s+)?senior applied scientist (role |position )?at meta",
        r"need for.{0,30}at meta",
        r"role of.{0,30}at meta",
        r"position at meta",
        r"aligns.{0,40}at meta",
        r"fit for.{0,40}at google",
        r"role.{0,20}at google",
    ]
    for pat in bad_patterns:
        m = re.search(pat, reply_lower)
        assert not m, \
            f"FAIL: Hallucination detected — output matches '{pat}'\n  Output: {reply}"

    # Check 2: reasoning references skills or gap counts (explainability)
    skill_signals = ["skill", "match", "missing", "required", "embeddings", "rag", "faiss",
                     "bge", "ndcg", "/20", "/7", "lacks", "without"]
    has_skill_ref = any(s in reply_lower for s in skill_signals)
    assert has_skill_ref, \
        "FAIL: Reasoning has no skill-level specifics — explainability still shallow"

    # Check 3: reasoning is NOT generic boilerplate
    generic_phrases = [
        "has experience in ml systems",
        "strong background in machine learning",
        "well-rounded candidate",
    ]
    for phrase in generic_phrases:
        assert phrase not in reply_lower, \
            f"FAIL: Generic boilerplate detected: '{phrase}'"

    print("PASS  [live] No company hallucination. Reasoning is skill-specific.")
    print(f"      Output:\n{reply}")


# ── Runner ────────────────────────────────────────────────────────────────────

def run_layer1():
    tests = [
        test_prompts_contain_target_role,
        test_reasoning_prompt_has_anti_hallucination_instruction,
        test_candidate_summary_includes_skill_gap,
        test_candidate_summary_without_required_skills_still_works,
        test_jd_parsing_extracts_redrob,
    ]
    failed = []
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed.append(t.__name__)

    print(f"\n{'='*55}")
    if failed:
        print(f"Layer 1: {len(tests) - len(failed)}/{len(tests)} passed. FAILED: {failed}")
        return False
    print(f"Layer 1: {len(tests)}/{len(tests)} passed. All prompt-level checks OK.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true",
                        help="Also run live LLM call (requires model on GPU)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct",
                        help="HuggingFace model for --live")
    args = parser.parse_args()

    ok = run_layer1()

    if args.live:
        print("\n" + "="*55)
        print("Layer 2: Live LLM spot-check")
        print("="*55)
        try:
            test_live_reasoning_does_not_hallucinate(args.model)
        except AssertionError as e:
            print(str(e))
            ok = False
        except Exception as e:
            print(f"ERROR: {e}")
            ok = False

    sys.exit(0 if ok else 1)
