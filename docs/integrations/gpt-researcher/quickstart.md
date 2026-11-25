# GPT Researcher Integration - Quick Start

🎯 **Goal:** Query NeurIPS 2025 presentations using GPT Researcher with persistent embeddings and entity tracking (no re-indexing!)

## TL;DR - Get Started in 3 Commands

```bash
# 1. Build the NeurIPS reference vector store (one-time historical path)
.venv/bin/python scripts/setup_vector_store_enhanced.py

# 2. Configure GPT Researcher (one-time)
.venv/bin/python scripts/setup_gpt_researcher_with_vectorstore.py

# 3. Query anytime (fast, unlimited queries!)
.venv/bin/python scripts/query_gpt_researcher_hybrid.py "What are the key innovations in autoregressive models?"

# 4. Check your generated report
ls conferences/neurips2025/cache/vector-store/reports/
```

---

## What This Does

This integration combines:

✅ **GPT Researcher** - Advanced AI research assistant
- Multi-iteration research
- Comprehensive report generation
- Intelligent source synthesis
- Structured outputs

✅ **ChromaDB Vector Store** - Persistent embeddings
- Index once, query unlimited times
- Fast semantic search
- Works across sessions
- Cost-effective

✅ **Tavily Web Search** - Real-time internet results
- Advanced search depth
- NeurIPS-contextual queries
- Complements local data with external sources

### Production Boundary

GPT-Researcher is the evidence-gathering and draft-finding layer. In current
production plans, local Qwen3.6 35B-A3B (`mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit`) plus
local BGE-M3 embeddings (`mlx-community/bge-m3-mlx-8bit`, 1024 dim) are the
default for corpus-grounded, local-only runs when receipts record
`WEB_SEARCH_ENABLED=false`, hybrid retrieval, source manifests, parameter
digests, and provider/model provenance. The final public
`analysis/<topic>.md` narrative remains governed
by `analysis.deep_narrative_synthesis` (`openai:gpt-5.6-sol`, fallback
`openai:gpt-5.6-terra`) unless a local candidate passes the same deep narrative
validation gates. Treat local GPT-Researcher reports as source-pack inputs, not
automatic production-final analysis.

### The Problem It Solves

**Before:** GPT Researcher re-created embeddings every single query
- ❌ Slow (5+ minutes per query)
- ❌ Repeated paid embedding work
- ❌ Not suitable for multiple queries

**After:** ChromaDB stores embeddings persistently
- ✅ Fast (~30 seconds per query)
- ✅ Reuses the one-time embedding index
- ✅ Perfect for unlimited queries across sessions

For current provider estimates, run:

```bash
.venv/bin/python -m conference_pipeline.costs estimate \
  --workspace conferences/neurips2025 \
  --scope gpt-researcher
```

Pricing comes from `src/conference_pipeline/costs/pricing_catalog.json`; final
costs come from OpenAI and Tavily receipts.

---

## Setup Options

### Option A: Production Workflow Setup (Recommended for New Conferences)

```bash
# Step 1: Build canonical source-aware vector stores
.venv/bin/python -m conference_pipeline.cli workflow run \
  --workspace conferences/<slug> \
  --stages enhanced-indexing-vector-store \
  --run-id <run-id>

# Step 2: Configure GPT Researcher (<1 min)
.venv/bin/python scripts/setup_gpt_researcher_with_vectorstore.py
```

Production runs use `src/conference_pipeline/indexing/chunking.py` through the
`enhanced-indexing-vector-store` workflow stage. They must build dual
`papers`/`artifacts` stores from full canonical chunk text, preserve source
locators, record chunk-length stats, and reject snippet-only or
first-240-character vectorization.

### Option B: Manual NeurIPS 2025 Reference Setup

```bash
# Step 1: Create enhanced vector store (~10-15 min)
.venv/bin/python scripts/setup_vector_store_enhanced.py

# Step 2: Configure GPT Researcher (<1 min)
.venv/bin/python scripts/setup_gpt_researcher_with_vectorstore.py
```

`scripts/setup_vector_store_enhanced.py` is pinned as the NeurIPS 2025
reference strategy that informed the canonical chunker. It is not the reusable
production owner command for future conferences.

### Option C: Automated Setup (Legacy Indexing)

```bash
./scripts/setup_all.sh
```

`setup_all.sh` uses `scripts/setup_vector_store.py` (legacy indexing). For new
conference production plans, use the workflow setup above; for historical
NeurIPS reproduction, use the pinned enhanced script.

---

## Usage Examples

### Data Location

The enhanced indexer expects NeurIPS 2025 data under `conferences/neurips2025/data`. Set `STORAGE_ROOT=conferences/neurips2025/data` in `.env` or move/symlink your data there.

### Entity Tracking Output

After indexing, review `conferences/neurips2025/cache/vector-store/entity_topic_index.json` to seed topic questions and synonym expansion.

### Basic Query (Hybrid: Local + Web)
```bash
.venv/bin/python scripts/query_gpt_researcher_hybrid.py "Summarize the main topics discussed at NeurIPS 2025"
```

### Specific Topic
```bash
.venv/bin/python scripts/query_gpt_researcher_hybrid.py "What are the latest developments in diffusion models?"
```

### Local Only (No Web Search)
```bash
.venv/bin/python scripts/query_gpt_researcher_hybrid.py --no-web "What AI agent frameworks were presented?"
```

### Web Only (External Sources)
```bash
.venv/bin/python scripts/query_gpt_researcher_hybrid.py --no-local "NeurIPS 2025 industry announcements"
```

### More Results Per Source
```bash
.venv/bin/python scripts/query_gpt_researcher_hybrid.py --max-results 20 "Compare approaches to scaling language models"
```

---

## Output

Each query generates a timestamped directory with multiple files:

```
conferences/neurips2025/cache/vector-store/reports/
└── 20251226_231926_what_ai_agent_frameworks_were_presented_/
    ├── report.md         # Final research report
    ├── sources.md        # Human-readable source list with relevance scores
    ├── sources.json      # Machine-readable sources
    └── raw_context.json  # All retrieved context chunks for inspection
```

### Report Structure
```markdown
# Research Report

**Query:** What are the latest developments in autoregressive models?
**Generated:** 2025-12-26T23:19:26
**Sources:** 30 total (15 local + 15 web)

---

## Executive Summary
[GPT-generated summary of key findings]

## Key Findings
[Detailed analysis with sources]

## References
[Full citations with URLs]
```

### Sources File (sources.md)
Shows all sources with relevance scores:
- **Local Sources**: NeurIPS presentations with timestamps and relevance %
- **Web Sources**: External articles with Tavily search scores

---

## Architecture

```
User Query
    ↓
GPT Researcher (query analysis)
    ↓
ChromaDB Vector Store (semantic search)
    ↓
Retrieved relevant chunks from 129 presentations
    ↓
GPT-4o (comprehensive analysis)
    ↓
Generated Report (markdown)
```

**Key Advantage:** ChromaDB embeddings persist on disk, so subsequent queries skip the embedding step entirely!

---

## File Structure

```
conferences/neurips2025/
└── vector_store/
    ├── chroma_db/                    # Persistent vector database ⭐
    │   ├── chroma.sqlite3
    │   └── [embeddings data]
    ├── gpt_researcher_config/
    │   ├── config.json
    │   └── chroma_retriever.py       # Custom retriever
    ├── reports/                      # Generated reports
    │   └── what_are_the_latest_developments_in_autor.md
    └── vector_store_metadata.json
```

---

## Troubleshooting

### "Vector store not found"
```bash
# Production conference run
.venv/bin/python -m conference_pipeline.cli workflow run \
  --workspace conferences/<slug> \
  --stages enhanced-indexing-vector-store \
  --run-id <run-id>

# Historical NeurIPS reference reproduction
.venv/bin/python scripts/setup_vector_store_enhanced.py
```

### "Custom retriever not found"
```bash
# Configure GPT Researcher
.venv/bin/python scripts/setup_gpt_researcher_with_vectorstore.py
```

### "OPENAI_API_KEY not found"
Live provider commands should load credentials from the repository-local `.env`
file before execution. Record only environment variable names in receipts and
docs, never secret values.

### Query is slow
First query loads the model (~30 sec). Subsequent queries are faster.

---

## Cost Breakdown

**One-time setup:**
- Creating embeddings: $0 paid API cost by default with local BGE-M3 MLX
  (`EMBEDDING_PROVIDER=mlx`, `EMBEDDING_DIM=1024`); OpenAI
  `text-embedding-3-small` is compatibility fallback only

**Per query:**
- Vector search: $0.00 (uses existing embeddings!)
- GPT-Researcher evidence/draft synthesis: $0 paid API cost by default with
  local Qwen3.6 35B-A3B; OpenAI/Tavily costs apply only when explicitly escalated

**Total cost for 100 local queries:** $0 paid API cost for GPT-Researcher
retrieval and draft synthesis, excluding hardware/runtime costs and any
explicit cloud escalation.

---

## Comparison with Other Approaches

| Approach | Setup | Query Speed | Cost/Query | Persistent | Best For |
|----------|-------|-------------|------------|------------|----------|
| Simple vector store | 5 min | Fast | $0.05 | ✅ Yes | Simple Q&A |
| GPT Researcher (original) | 1 min | Slow | $1.00+ | ❌ No | One-off research |
| **GPT Researcher + ChromaDB** ⭐ | 10 min | Fast | $0.05 | ✅ Yes | **Production** |

---

## Advanced Configuration

For the reusable workflow stage, configure
`workflow.research.gpt_researcher` in the conference `config.yaml`:

```json
{
  "retrieval_mode": "hybrid",
  "max_iterations": 5,
  "deep_research_breadth": 4,
  "deep_research_depth": 3,
  "max_subtopics": 8,
  "total_words": 3000,
  "smart_token_limit": 8192,
  "strategic_token_limit": 8192,
  "fast_token_limit": 4096,
  "retrieval_k": 18,
  "paper_matching_retrieval_k": 18,
  "candidate_multiplier": 3,
  "rrf_k": 60,
  "temperature": 0.0
}
```

One-off runs can also use uppercase env aliases accepted by upstream
GPT-Researcher, for example `MAX_ITERATIONS=5`,
`DEEP_RESEARCH_BREADTH=4`, or `RETRIEVAL_K=18`. The production stage records the resolved
parameters in query-plan JSON and the stage receipt. The behavioral resume
digest currently includes only knobs that change this stage's execution or
synthesis prompt; `DEEP_RESEARCH_CONCURRENCY`, `FAST_TOKEN_LIMIT`, and
`STRATEGIC_TOKEN_LIMIT` are recorded for parity but are receipt-only until
parallel sub-research or separate fast/strategic calls are implemented.

---

## Next Steps

1. For new conference production, run `enhanced-indexing-vector-store` and `scripts/setup_gpt_researcher_with_vectorstore.py`; for historical NeurIPS reproduction, run `scripts/setup_vector_store_enhanced.py`.
2. Try your first query
3. Check the generated report in `conferences/neurips2025/cache/vector-store/reports/`
4. Experiment with different question types
5. Share insights from the conference data

---

## Full Documentation

- **[Integration Guide](integration-guide.md)** - Complete guide with all features
- **[Vector Store Guide](vector-store-guide.md)** - Comparison of all approaches

---

## Scripts Reference

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `setup_all.sh` | Automated setup (legacy indexing) | Quick setup |
| `setup_vector_store_enhanced.py` | Reproduce the pinned enhanced ChromaDB reference strategy | Historical NeurIPS 2025 analysis workflow |
| `conference_pipeline.indexing.chunking` | Canonical source-aware chunk builders used by `enhanced-indexing-vector-store` | Reusable production conference workflow |
| `setup_vector_store.py` | Create legacy index | Backwards compatibility |
| `setup_gpt_researcher_with_vectorstore.py` | Configure integration | After creating vector store |
| `query_gpt_researcher_hybrid.py` | Query with GPT Researcher | Every time you want to research |
| `query_vector_store.py` | Simple vector search | Quick lookups without full reports |

---

**Ready to get started? Run the enhanced setup commands above.**

## Source References

- scripts/setup_vector_store_enhanced.py
- scripts/setup_gpt_researcher_with_vectorstore.py
- scripts/query_gpt_researcher_hybrid.py
- scripts/query_vector_store.py
