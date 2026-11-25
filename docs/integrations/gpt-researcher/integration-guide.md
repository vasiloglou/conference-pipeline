# GPT Researcher + ChromaDB Integration Guide

This guide explains how to use GPT Researcher with a persistent ChromaDB vector store, combining the best of both worlds.

## What You Get

✅ **GPT Researcher's Advanced Features:**
- Comprehensive multi-source research reports
- Intelligent sub-topic generation
- Structured report formats (research reports, outlines, resource lists)
- Multi-iteration research refinement

✅ **Persistent Vector Store Benefits:**
- **Index once, query unlimited times** (no re-embedding!)
- Fast semantic search across all presentations
- Works across different sessions
- Cost-effective (embeddings created once)

---

## Setup Process

### Step 1: Create the Enhanced Vector Store (One-Time, ~10-15 minutes)

First, create the ChromaDB vector store with enhanced chunking and entity tracking:

```bash
.venv/bin/python scripts/setup_vector_store_enhanced.py
```

**Data location:** The enhanced indexer expects NeurIPS 2025 data under `conferences/neurips2025/data`. Set `STORAGE_ROOT=conferences/neurips2025/data` in `.env` or move/symlink your data there.

This will:
- Process all 129 presentations and 9,825 papers
- Chunk transcripts using `segments.json` (preserves timestamps)
- Index slides individually with timestamps
- Index paper title + abstract together
- Create embeddings using OpenAI's `text-embedding-3-small`
- Store everything in `conferences/neurips2025/cache/vector-store/chroma_db/`
- Generate `conferences/neurips2025/cache/vector-store/entity_topic_index.json`

**Expected output:**
```
============================================================
Vector Store Setup for Conference Pipeline
============================================================

Step 1: Loading environment variables...
✓ Loaded OPENAI_API_KEY
✓ Loaded TAVILY_API_KEY

Step 2: Loading presentation data...
✓ Loaded 129 presentations

Step 3: Creating persistent vector store...
(This will take a few minutes - creating embeddings...)
Processing [1/129]: presentation-name
...
✓ Created vector store with ~X chunks
```

**Time:** 10-15 minutes (creates embeddings for presentations + papers)
**Cost:** Estimate with the canonical registry:
```bash
.venv/bin/python -m conference_pipeline.costs estimate \
  --workspace conferences/neurips2025 \
  --scope gpt-researcher
```

---

### Step 2: Configure GPT Researcher to Use the Vector Store (One-Time, <1 minute)

Configure GPT Researcher to use the ChromaDB backend:

```bash
.venv/bin/python scripts/setup_gpt_researcher_with_vectorstore.py
```

This will:
- Create GPT Researcher configuration
- Set up custom ChromaDB retriever
- Link everything together

**Expected output:**
```
============================================================
GPT Researcher + ChromaDB Integration Setup
============================================================

✓ Loaded OPENAI_API_KEY
✓ Loaded TAVILY_API_KEY

Step 1: Creating GPT Researcher configuration...
✓ Created GPT Researcher config at: ...

Step 2: Creating custom ChromaDB retriever...
✓ Created custom ChromaDB retriever at: ...

============================================================
Setup Complete!
============================================================
```

---

## Usage

### Query Anytime (Fast - Uses Existing Embeddings!)

Once setup is complete, you can query unlimited times:

```bash
.venv/bin/python scripts/query_gpt_researcher_hybrid.py "Your research question"
```

**Example queries:**

```bash
# Comprehensive research on a topic
.venv/bin/python scripts/query_gpt_researcher_hybrid.py "What are the key innovations in autoregressive models presented at NeurIPS 2025?"

# Comparative analysis
.venv/bin/python scripts/query_gpt_researcher_hybrid.py "Compare different transformer architectures discussed at the conference"

# Application-specific
.venv/bin/python scripts/query_gpt_researcher_hybrid.py "How is reinforcement learning being applied to robotics?"

# Technical deep-dive
.venv/bin/python scripts/query_gpt_researcher_hybrid.py "What are the latest developments in scaling laws for large language models?"
```

### What Happens During a Query

1. **Connects to existing vector store** (no re-indexing!)
2. **Semantic search** finds most relevant presentation chunks
3. **GPT Researcher analyzes** the context
4. **Generates comprehensive report** with:
   - Executive summary
   - Key findings organized by topic
   - Sources and references
   - Detailed analysis
5. **Saves report** to `conferences/neurips2025/cache/vector-store/reports/`

**Time per query:** 30-60 seconds (depending on query complexity)
**Cost per query:** Use the estimator output for GPT Researcher LLM calls,
embedding refreshes, and Tavily credits. The static registry lives at
`src/conference_pipeline/costs/pricing_catalog.json`; provider receipts are
authoritative after a run.

---

## Report Types

You can modify `query_gpt_researcher_hybrid.py` to use different report types:

```python
# In the script, change the report_type parameter:
report, context = await research_with_chromadb(
    query,
    report_type="research_report"  # Options below
)
```

**Available report types:**
- `research_report` - Comprehensive research report (default)
- `outline_report` - Structured outline format
- `resource_report` - Curated list of resources
- `custom_report` - Custom format based on your prompt
- `subtopic_report` - Multiple subtopics with deep analysis

---

## Comparison: Three Approaches

| Feature | Simple Vector Store | GPT Researcher (Original) | **GPT Researcher + ChromaDB** ⭐ |
|---------|-------------------|-------------------------|------------------------------|
| Setup time | 5-10 min | 1 min | 6-11 min |
| Persistent index | ✅ Yes | ❌ No | ✅ Yes |
| Re-indexes each query | ❌ No | ✅ Yes | ❌ No |
| Query speed | Fast | Slow | Fast |
| Report quality | Basic | Excellent | Excellent |
| Multi-iteration research | ❌ No | ✅ Yes | ✅ Yes |
| Sub-topic generation | ❌ No | ✅ Yes | ✅ Yes |
| Cost per query | Low | High | Low |
| **Recommended for** | Simple Q&A | One-off deep research | **Production use** |

---

## Directory Structure

After setup, you'll have:

```
conferences/neurips2025/cache/vector-store/
├── chroma_db/                    # Persistent ChromaDB database
│   ├── chroma.sqlite3            # Vector database
│   └── ...                       # Embedding data
├── gpt_researcher_config/        # GPT Researcher configuration
│   ├── config.json               # Settings
│   └── chroma_retriever.py       # Custom retriever
├── entity_topic_index.json       # Entity tracking output
├── reports/                      # Generated reports
│   └── *.md                      # Individual query reports
├── queries/                      # Simple query results (from basic vector store)
│   └── *.md
└── vector_store_metadata.json   # Index metadata
```

---

## Troubleshooting

### "Vector store not found"
Run step 1 first:
```bash
.venv/bin/python scripts/setup_vector_store_enhanced.py
```

### "Custom retriever not found"
Run step 2 first:
```bash
.venv/bin/python scripts/setup_gpt_researcher_with_vectorstore.py
```

### "OPENAI_API_KEY not found"
Make sure the repository's `.env` contains:
```bash
OPENAI_API_KEY="your-key-here"
```

Then load it into the shell before running the script: `set -a; source .env; set +a`

### Slow queries
This is normal for the first query (loading the model). Subsequent queries should be much faster.

---

## Re-indexing

If you add new presentations or want to rebuild the index:

```bash
# Delete old vector store
rm -rf conferences/neurips2025/cache/vector-store/chroma_db

# Recreate
.venv/bin/python scripts/setup_vector_store_enhanced.py

# Reconfigure (if needed)
.venv/bin/python scripts/setup_gpt_researcher_with_vectorstore.py
```

---

## Advanced: Multiple Collections

You can create separate vector stores for different conferences:

```python
# In setup_vector_store_enhanced.py, change:
collection_name = "icml_2025_presentations"  # Instead of "neurips_presentations"
```

Then query specific collections by modifying the retriever initialization.

---

## Quick Start Summary

```bash
# 1. Create vector store (one-time, ~10-15 min)
.venv/bin/python scripts/setup_vector_store_enhanced.py

# 2. Configure GPT Researcher (one-time, <1 min)
.venv/bin/python scripts/setup_gpt_researcher_with_vectorstore.py

# 3. Query anytime (unlimited, fast!)
.venv/bin/python scripts/query_gpt_researcher_hybrid.py "Your question"
```

That's it! You now have a production-ready research system that combines GPT Researcher's intelligence with persistent embeddings.

## Source References

- scripts/setup_vector_store_enhanced.py
- scripts/setup_gpt_researcher_with_vectorstore.py
- scripts/query_gpt_researcher_hybrid.py
