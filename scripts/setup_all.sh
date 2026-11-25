#!/bin/bash

# Complete setup script for GPT Researcher + ChromaDB integration

echo "============================================================"
echo "Complete Setup: GPT Researcher + ChromaDB Vector Store"
echo "============================================================"
echo ""
echo "This will set up everything you need to query the conference"
echo "presentations using GPT Researcher with persistent embeddings."
echo ""
echo "Total time: ~10-15 minutes"
echo "Cost: ~$0.50-1.50 in OpenAI API calls (one-time)"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Setup cancelled."
    exit 0
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo ""
echo "Step 1/2: Creating enhanced ChromaDB vector store..."
echo "This will take 10-15 minutes and create embeddings for presentations + papers."
echo ""

cd "$PROJECT_ROOT"
.venv/bin/python scripts/setup_vector_store_enhanced.py

if [ $? -ne 0 ]; then
    echo ""
    echo "Error: Vector store setup failed!"
    exit 1
fi

echo ""
echo "Step 2/2: Configuring GPT Researcher..."
echo ""

.venv/bin/python scripts/setup_gpt_researcher_with_vectorstore.py

if [ $? -ne 0 ]; then
    echo ""
    echo "Error: GPT Researcher configuration failed!"
    exit 1
fi

echo ""
echo "============================================================"
echo "✅ Complete Setup Finished!"
echo "============================================================"
echo ""
echo "You can now query the conference data with:"
echo ""
echo "  .venv/bin/python scripts/query_gpt_researcher_hybrid.py \"Your question\""
echo ""
echo "Example:"
echo "  .venv/bin/python scripts/query_gpt_researcher_hybrid.py \"What are the latest developments in autoregressive models?\""
echo ""
echo "See docs/integrations/gpt-researcher/integration-guide.md for full documentation."
echo ""
