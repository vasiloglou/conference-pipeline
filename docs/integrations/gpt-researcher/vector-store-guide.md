# Vector Store Guide

This project supports **two approaches** for querying conference presentation
evidence. The reusable production path now builds persistent vector stores with
canonical source-aware chunking in `src/conference_pipeline/indexing/chunking.py`.
The older NeurIPS 2025 `scripts/setup_vector_store_enhanced.py` script is kept
as the pinned reference strategy that inspired the reusable chunker; it is not
the production owner command for future conferences.

## Approach 1: GPT Researcher (Re-indexes each time)

**Location:** `conferences/neurips2025/agent-work/gpt-researcher/`

### How it works:
- Uses GPT Researcher library
- Reads text documents from the `documents/` directory
- **Re-processes and creates embeddings on each query** (slower, more expensive)
- Good for: Quick setup, testing, one-off queries

### Setup (Already Done):
```bash
.venv/bin/python scripts/setup_gpt_researcher.py
```

### Query:
```bash
.venv/bin/python scripts/query_gpt_researcher.py "Your question here"
```

### Pros:
- Simple setup
- Full GPT Researcher features (web search, report generation)

### Cons:
- ⚠️ **Re-creates embeddings every time** (slow and costs API credits)
- Not suitable for multiple queries in different sessions

---

## Approach 2: Persistent Vector Store (Recommended)

**Location:** `conferences/neurips2025/cache/vector-store/`

### How it works:
- Uses ChromaDB for persistent vector storage
- **Creates embeddings once** during setup
- **Reuses embeddings** across all future queries (fast, cost-effective)
- Vector database stored on disk at `vector_store/chroma_db/`
- Good for: Production use, multiple queries, repeated access

### Setup (One-time):

For production conference runs, build the store through the workflow stage:

```bash
.venv/bin/python -m conference_pipeline.cli workflow run \
  --workspace conferences/<slug> \
  --stages enhanced-indexing-vector-store \
  --run-id <run-id>
```

That stage calls the canonical chunker and writes reusable `papers` and
`artifacts` stores for dual-store GPT-Researcher / paper matching. Production
vector stores must use full chunk text, preserve source locators, record
chunk-length statistics, and reject snippet-only or first-240-character
vectorization.

For historical NeurIPS 2025 reproduction only:

```bash
# Install ChromaDB if not already installed
.venv/bin/pip install chromadb

# Reproduce the pinned NeurIPS reference vector store (takes 10-15 minutes)
.venv/bin/python scripts/setup_vector_store_enhanced.py
```

The reference script will:
1. Load all 129 presentations and 9,825 papers
2. Create transcript chunks from `segments.json` (preserves timestamps)
3. Index slides individually with timestamps
4. Index paper title + abstract together
5. Generate embeddings using OpenAI's `text-embedding-3-small`
6. Store everything in a persistent ChromaDB database
7. Save metadata and entity tracking output

**Data location:** The NeurIPS reference script expects NeurIPS 2025 data under
`conferences/neurips2025/data`. Set
`STORAGE_ROOT=conferences/neurips2025/data` or move/symlink your data there.

### Query (Can be run unlimited times):
```bash
.venv/bin/python scripts/query_vector_store.py "What are the latest developments in autoregressive models?"
```

Each query:
1. Connects to the existing vector store (no re-indexing!)
2. Searches for relevant chunks using semantic similarity
3. Retrieves top results
4. Uses GPT-4o to generate a comprehensive answer with sources
5. Saves the result to `vector_store/queries/`

### Pros:
- ✅ **Much faster** (no re-indexing)
- ✅ **Cost-effective** (embeddings created once)
- ✅ **Persistent** (works across multiple sessions)
- ✅ **Semantic search** with similarity scores
- ✅ **Production-ready**

### Cons:
- Requires one-time setup
- Takes ~10-15 minutes to create initial index

---

## Comparison

| Feature | GPT Researcher | Vector Store |
|---------|---------------|--------------|
| Setup time | 1 min | 10-15 min |
| First query | Slow (creates embeddings) | Fast |
| Subsequent queries | Slow (re-creates embeddings) | Fast |
| Persistence | ❌ No | ✅ Yes |
| Cost per query | High (re-embeds everything) | Low (reuses embeddings) |
| Recommended for | Testing, one-off | Production, multiple queries |

---

## Recommendation

**For your use case (querying multiple times in different sessions):**

👉 **Use the Persistent Vector Store (Approach 2)**

1. Run the workflow setup once for a production conference:
   ```bash
   .venv/bin/python -m conference_pipeline.cli workflow run \
     --workspace conferences/<slug> \
     --stages enhanced-indexing-vector-store \
     --run-id <run-id>
   ```

   For historical NeurIPS reproduction only:
   ```bash
   .venv/bin/python scripts/setup_vector_store_enhanced.py
   ```

2. Then query as many times as you want:
   ```bash
   .venv/bin/python scripts/query_vector_store.py "Your question"
   ```

The vector store will persist on disk and can be reused indefinitely without re-indexing!

---

## Example Queries

```bash
# Research specific topics
.venv/bin/python scripts/query_vector_store.py "What innovations in autoregressive models were presented?"

# Compare approaches
.venv/bin/python scripts/query_vector_store.py "Compare different transformer architectures discussed"

# Find applications
.venv/bin/python scripts/query_vector_store.py "How is AI being applied to robotics?"

# Scaling laws
.venv/bin/python scripts/query_vector_store.py "What do presentations say about scaling laws?"
```

---

## Vector Store Structure

```
conferences/neurips2025/cache/vector-store/
├── chroma_db/              # Persistent ChromaDB database
│   ├── chroma.sqlite3      # SQLite database
│   └── ...                 # Embedding data
├── entity_topic_index.json # Entity tracking output
├── queries/                # Saved query results
│   └── *.md                # Individual query outputs
└── vector_store_metadata.json  # Index metadata
```

---

## Re-indexing

If you add new presentations or want to rebuild the index:

```bash
# Delete old NeurIPS reference vector store
rm -rf conferences/neurips2025/cache/vector-store/chroma_db

# Re-run historical reference setup
.venv/bin/python scripts/setup_vector_store_enhanced.py
```

---

## API Keys

Both approaches use:
- `OPENAI_API_KEY` - For embeddings and LLM
- `TAVILY_API_KEY` - For web search (GPT Researcher only)

Live provider commands should load credentials from the repository-local `.env`
file before execution; receipts and docs must record only variable names, never
secret values.

## Source References

- src/conference_pipeline/indexing/chunking.py
- src/conference_pipeline/indexing/walker.py
- scripts/setup_vector_store_enhanced.py (pinned NeurIPS reference strategy)
- scripts/setup_gpt_researcher.py
- scripts/setup_gpt_researcher_with_vectorstore.py
- scripts/query_vector_store.py
- scripts/query_gpt_researcher_hybrid.py
