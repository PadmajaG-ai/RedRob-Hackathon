# Hackathon: AI Candidate Ranking System

## Project Context

Hackathon Track 1 — Build an intelligent AI-powered candidate ranking system that goes beyond keyword filters and surfaces the right candidates for a role.

**Judging criteria:**
- Quality of ranking output
- Clarity of methodology and reasoning
- Explainability of the system

**Dataset status:** Not yet received. Expected in ~2 days. When it arrives, first check:
- Schema/format of resumes and profiles
- Behavioral signal structure (raw logs vs. pre-aggregated)
- Any ground truth (past hires, ratings) — unlocks learned fusion and evaluation metrics
- Scale — number of candidates per JD (affects latency decisions)

---

## System Specs (Linux Machine)

- RAM: 256GB
- GPU: NVIDIA 24GB VRAM
- This machine runs the full pipeline locally — no external API dependencies needed except for one-time synthetic data generation via Claude API.

---

## Architecture: Two-Stage Retrieval + Re-ranking Pipeline

```
JD → [JD Analyzer] → Structured Criteria
         ↓
Candidates → [Parser] → Enriched Profiles
         ↓
[Stage 1: bge-m3 Hybrid Retrieval]   ← fast, high recall
   FAISS vector store
         ↓
   Top 50 candidates
         ↓
[Stage 2: bge-reranker-v2-m3 + LoRA]  ← domain-adapted reranker
         ↓
[LLM-as-Judge: Llama-3-8B-Instruct via vLLM]  ← multi-dimensional scoring
         ↓
   Final Ranked Shortlist + Per-candidate Reasoning
```

---

## Finalized Tech Stack

| Component | Model / Tool | Reason |
|---|---|---|
| Parsing | `PyMuPDF` + `spaCy` | Fast, local, no server needed |
| Embeddings | `bge-m3` | Single model: dense + sparse + multi-vector — no separate BM25 index needed |
| Vector store | `FAISS` | Local, no server, pip install |
| Reranker | `bge-reranker-v2-m3` + LoRA adapter | Best open reranker, supports LoRA plugin architecture |
| LLM Judge | `Llama-3-8B-Instruct` via `vLLM` | Batched inference, scores 50 candidates in ~15-20s |
| Fine-tuning | `peft` + `bitsandbytes` (QLoRA) | Runs in 1-2 hours on 24GB GPU |
| Synthetic data | Claude API (one-time use) | Generate (JD, good_profile, bad_profile) triplets for fine-tuning |

---

## Key Architecture Decisions (with Reasoning)

### Why Two-Stage (Retrieval + Re-ranking)?
- Stage 1 optimizes for **recall** — don't miss good candidates
- Stage 2 optimizes for **precision** — deep contextual scoring on shortlist only
- Running LLM-as-judge on all candidates is too slow; running it only on top-50 is fast

### Why bge-m3 over separate dense + BM25?
- Single model outputs dense vector + learned sparse vector + multi-vector (ColBERT-style)
- Learned sparse is better than BM25 — understands "ML engineer" = "machine learning"
- No RRF fusion needed between separate retrievers; bge-m3 handles it internally
- RRF (Reciprocal Rank Fusion) is rank-based and treats all retrievers equally — bge-m3 avoids this limitation

### Why LLM-as-Judge over Cross-Encoder only?
- Cross-encoders (Option A) produce a single relevance score — no dimension breakdown
- LLM-as-Judge (Option B) scores each dimension separately with reasoning — directly satisfies explainability criteria
- Used on top-50 shortlist only, so latency is acceptable

### Why vLLM over Ollama?
- vLLM uses continuous batching — processes 50 candidates in parallel
- Ollama processes sequentially: 50 × 3s = 2.5 min vs vLLM ~15-20s
- Critical for demo speed

### Why LoRA fine-tune the reranker?
- Off-the-shelf rerankers trained on web search, not recruiting
- LoRA adapter on synthetic recruiting pairs teaches domain-specific relevance
- "Led a team of 5" → relevant for "people management required" even without keyword overlap
- Adapter files are small (~50MB), training takes 1-2 hours on this GPU

---

## Scoring Dimensions (LLM-as-Judge Rubric)

Score each candidate 0–10 on:
1. **Technical skill match** — depth + recency of required skills
2. **Career trajectory alignment** — seniority, progression speed, growth rate
3. **Domain/industry relevance** — industry verticals, company size, product vs. services
4. **Behavioral engagement signals** — platform activity, contributions, endorsements
5. **Cultural/soft skill fit** — signals from JD language vs. candidate communication style
6. **Role-specific bonus signals** — e.g., open source for engineering, publications for research

Final score = weighted sum across dimensions + penalty for missing hard requirements.
Weights are configurable per role type (IC vs. lead vs. manager).

---

## Candidate Profile Enrichment (Signal Types)

| Signal Type | Examples |
|---|---|
| Career trajectory | Progression speed, lateral moves, gaps, tenure patterns |
| Skill depth | Years used, recency, context (solo vs. team) |
| Domain experience | Industry verticals, company size, product vs. services |
| Behavioral signals | Platform activity, contribution frequency, peer endorsements |
| Metadata | Location, availability, seniority self-assessment |

Career *progression* often matters more than current title.

---

## Implementation Roadmap (Once Dataset Arrives)

### Day 1
- [ ] Parse and explore dataset schema (2 hours)
- [ ] Build JD analyzer — extract weighted criteria tree using LLM
- [ ] Build candidate profile parser and enrichment pipeline
- [ ] Set up FAISS + bge-m3 retrieval baseline
- [ ] Generate synthetic (JD, good_profile, bad_profile) triplets via Claude API (~500-1000 pairs)
- [ ] QLoRA fine-tune bge-reranker-v2-m3 on synthetic pairs (1-2 hours on GPU)

### Day 2
- [ ] Evaluate baseline reranker vs. fine-tuned reranker — document the delta
- [ ] Set up vLLM serving for Llama-3-8B-Instruct
- [ ] Build LLM-as-Judge scoring with structured JSON output (dimension scores + reasoning)
- [ ] Build signal fusion layer (weighted composite score + penalty system)
- [ ] Build final output: ranked shortlist + per-candidate explanation
- [ ] End-to-end test, tune weights, prepare demo

---

## Differentiators vs. Other Teams

1. **Domain-adapted retrieval** — LoRA fine-tuned reranker on recruiting pairs, not off-the-shelf
2. **Fully local LLM judge** — no OpenAI/API dependency, structured multi-dimensional scoring
3. **bge-m3 unified retrieval** — dense + sparse + multi-vector from one model
4. **Explainability built-in** — every rank position has a human-readable justification per dimension
5. **Contextual profile enrichment** — career trajectory and behavioral signals, not just keyword matching

---

## Contextual Retrieval Note

Before embedding candidate profile chunks, prepend LLM-generated context:
```
Original:      "Led migration to microservices architecture"
Contextualized: "Candidate with 8 years backend experience at Series B fintech.
                 Led migration to microservices architecture."
```
This significantly improves retrieval because vectors capture *who did what and where*, not just raw text.
Anthropic reported ~49% reduction in retrieval failures with this technique.
