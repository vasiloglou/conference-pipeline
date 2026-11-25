#!/bin/bash
# Run fact ranking and paper ranking for all 20 topics
# Usage: ./scripts/run_all_rankings.sh

set -e

cd "$(dirname "$0")/.."

ANALYSIS_DIR="conferences/neurips2025/analysis"
PODCASTS_DIR="conferences/neurips2025/audio"
CLAUDE_REPORTS="conferences/neurips2025/agent-work/claude/paper-influence/reports"
CODEX_REPORTS="conferences/neurips2025/agent-work/codex/paper-influence/reports"

# Topics to process
TOPICS=(
    "01_tools_and_open_source"
    "02_genai_architectures"
    "03_data_preparation"
    "04_data_attribution"
    "05_data_markets"
    "06_visual_models"
    "07_tabular_models"
    "08_language_models"
    "09_genai_in_science"
    "10_serendipitous_findings"
    "11_ethical_problems"
    "12_commercial_applications"
    "13_industry_insights"
    "14_synthetic_data"
    "15_whats_happening_at_neurips"
    "16_neurips_awards"
    "17_llm_scaling_laws_efficiency"
    "18_agentic_ai_autonomous_systems"
    "19_notable_quotes"
    "20_small_language_models"
)

echo "=========================================="
echo "PHASE 1: FACT RANKING (all 20 topics)"
echo "=========================================="

for topic in "${TOPICS[@]}"; do
    echo ""
    echo ">>> Processing: $topic"
    echo "-------------------------------------------"

    ANALYSIS_FILE="${ANALYSIS_DIR}/${topic}.md"
    OUTPUT_DIR="${PODCASTS_DIR}/${topic}"

    if [ ! -f "$ANALYSIS_FILE" ]; then
        echo "  SKIP: Analysis file not found: $ANALYSIS_FILE"
        continue
    fi

    # Create output directory
    mkdir -p "$OUTPUT_DIR"

    # Check if fact ranking already done
    if [ -f "${OUTPUT_DIR}/ranks.json" ]; then
        echo "  SKIP: Fact ranking already exists"
    else
        echo "  Running fact ranking (hybrid, 10 frequency runs)..."
        .venv/bin/python scripts/rank_facts.py \
            "$ANALYSIS_FILE" \
            --method hybrid \
            --n-runs 10 \
            --output-dir "$OUTPUT_DIR" \
            --model gemini-2.0-flash || echo "  WARNING: Fact ranking failed for $topic"
    fi
done

echo ""
echo "=========================================="
echo "PHASE 2: PAPER RANKING (all 20 topics)"
echo "=========================================="

for topic in "${TOPICS[@]}"; do
    echo ""
    echo ">>> Processing: $topic"
    echo "-------------------------------------------"

    OUTPUT_DIR="${PODCASTS_DIR}/${topic}/paper_ranks"
    ANALYSIS_FILE="${ANALYSIS_DIR}/${topic}.md"

    # Check if paper ranking already done
    if [ -f "${OUTPUT_DIR}/ranked_papers.json" ]; then
        echo "  SKIP: Paper ranking already exists"
        continue
    fi

    # Check if reports exist for this topic
    CLAUDE_REPORT="${CLAUDE_REPORTS}/${topic}_influential_papers.md"
    CODEX_REPORT="${CODEX_REPORTS}/${topic}_influential_papers.md"

    if [ ! -f "$CLAUDE_REPORT" ] && [ ! -f "$CODEX_REPORT" ]; then
        echo "  SKIP: No paper reports found for $topic"
        continue
    fi

    echo "  Running paper ranking (5 frequency runs)..."
    .venv/bin/python scripts/rank_papers.py \
        --topic "$topic" \
        --claude-reports "$CLAUDE_REPORTS" \
        --codex-reports "$CODEX_REPORTS" \
        --analysis-doc "$ANALYSIS_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --frequency-runs 5 \
        --target-minutes 10 \
        --model gemini-2.0-flash || echo "  WARNING: Paper ranking failed for $topic"
done

echo ""
echo "=========================================="
echo "COMPLETE!"
echo "=========================================="
echo "Fact rankings: ${PODCASTS_DIR}/*/ranks.json"
echo "Paper rankings: ${PODCASTS_DIR}/*/paper_ranks/ranked_papers.json"
