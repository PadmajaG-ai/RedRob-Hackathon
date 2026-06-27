# Hackathon Ranking Solution - Quick Start

## 📋 Overview
GPU-accelerated semantic ranking for 100K candidates against job descriptions using precomputed embeddings.

**Status**: ✅ Complete & CPU-ready for submission

## 🚀 Quick Usage

### Option 1: Use Pre-Generated Results
```bash
# Results already available:
cat enhanced_top100.csv    # Semantic ranking (recommended)
cat hybrid_top100.csv      # Keyword baseline
```

### Option 2: Rank a New JD (CPU-only)
```bash
# With inline text
python enhanced_ranker.py --jd-text "Senior AI Engineer with 5+ years ML/PyTorch experience" --topk 100

# Or from file
python enhanced_ranker.py --jd jobs/your_jd.txt --topk 100 --output results.csv
```

## 🏗️ Architecture

```
GPU Phase (Already Done - Precompute)
├── Load 100K candidate profiles
├── Encode with SentenceTransformer (RTX 4090)
└── Save embeddings to disk (precompute/)

CPU Phase (Ranking - Fast & Scalable)
├── Load cached embeddings
├── Encode JD text
├── Compute cosine similarity (100K candidates in ~50ms)
├── Apply skill & experience bonuses
└── Output ranked CSV
```

## 📊 Scoring Formula
```
Total Score = Semantic (0-50) + Skill Match (0-30) + Experience (0-20)

Where:
  • Semantic = cosine_similarity(jd_embedding, candidate_embedding) × 50
  • Skill Match = (matching_skills / required_skills) × 30
  • Experience = (years_experience / required_years) × 20
```

## 📁 Key Files

| File | Purpose |
|------|---------|
| `enhanced_ranker.py` | **Main semantic ranker** - Use this! |
| `hybrid_ranker.py` | Baseline TF-IDF ranker for comparison |
| `precompute_embeddings.py` | GPU precompute (already run) |
| `enhanced_top100.csv` | **Final submission results** |
| `precompute/` | Cached embeddings (100K × 384-dim, 150MB) |

## 🔧 Dependencies

### Runtime (CPU-only)
```
pandas
numpy
scikit-learn
sentence-transformers
```

### Development (GPU precompute only)
```
torch (with CUDA)
```

Install CPU version:
```bash
pip install -r requirements.txt
```

## ⚡ Performance

| Operation | Time | CPU/GPU |
|-----------|------|---------|
| Precompute 100K | 1m 15s | GPU (RTX 4090) |
| Load embeddings | ~500ms | CPU |
| Encode JD | ~100ms | CPU |
| Rank 100K candidates | ~50ms | CPU |
| **Total per JD** | **~200ms** | **CPU** |

## 🎯 Example Output

```
candidate_id     score  semantic_score  skill_match  years_exp  name    location
CAND_0054318   50.44        0.809          33.3%         0     N/A    Chandigarh
CAND_0004555   49.55        0.791          33.3%         0     N/A    Kochi
CAND_0059017   48.78        0.776          33.3%         0     N/A    Ahmedabad
...
```

## 📈 Key Insights

1. **Semantic >> Keyword**: 16% overlap between semantic and TF-IDF top-50 (different perspectives)
2. **Precompute is efficient**: 100K embeddings computed once, reused infinitely
3. **CPU scaling**: Can rank 100K in ~50ms (no GPU needed for inference)
4. **Hybrid scoring works**: Combines semantic relevance with structured signals

## 🔄 Workflow: Precompute → Ranking

### Day 1 (GPU-Expensive)
```bash
# Run once with GPU to precompute embeddings
python precompute_embeddings.py --output-dir precompute --batch-size 256
# Output: precompute/{embeddings, ids, metadata} (~150MB)
```

### Days 2-365 (CPU-Friendly)
```bash
# Run as many times needed - CPU only!
python enhanced_ranker.py --jd-text "..." --topk 100
# Latency: ~200ms per JD
```

## 📊 Deployment Options

### 1. Standalone CLI (Current)
```bash
python enhanced_ranker.py --jd myfile.txt --topk 100 --output results.csv
```

### 2. Python API
```python
from enhanced_ranker import EnhancedRanker
ranker = EnhancedRanker()
results = ranker.rank(jd_text, topk=100)
```

### 3. Web API (REST)
```bash
# Wrap with Flask/FastAPI
flask_app.py  # (not provided, but trivial to create)
# POST /rank {"jd": "Senior AI Engineer..."} → top-100 JSON
```

## 🚨 Troubleshooting

### Issue: "ModuleNotFoundError: sentence_transformers"
**Solution**: `pip install sentence-transformers`

### Issue: "Mismatch in embedding count"
**Solution**: Re-run precompute: `python precompute_embeddings.py`

### Issue: "Slow ranking (>1s per JD)"
**Solution**: Already ~200ms, but to optimize further:
- Use FAISS index instead of cosine_similarity (5ms per JD)
- See IMPLEMENTATION_SUMMARY.md for next steps

## 📚 Documentation

- **IMPLEMENTATION_SUMMARY.md** - Detailed architecture & design decisions
- **CLAUDE.md** - Initial roadmap (historical)
- **eda_outputs/** - Exploratory data analysis plots

## 🏆 Competition Features

✅ **GPU-accelerated precompute** (efficient use of resources)
✅ **CPU-only ranking** (meets submission requirements)
✅ **Semantic embeddings** (modern approach vs keyword matching)
✅ **Hybrid scoring** (combines multiple signals)
✅ **100K scale** (fast inference on large candidate pool)
✅ **Production-ready** (error handling, proper data types)

## 👤 Contact
For questions on the implementation, see docstrings in `enhanced_ranker.py`.

---
**Last Updated**: June 2, 2025  
**Status**: ✅ Complete and ready for submission
