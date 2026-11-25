#!/bin/bash
# Query Gemini using existing Chrome via CDP connection

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Your Gemini URL with account selector
GEMINI_URL="https://gemini.google.com/u/2/app?pageId=none"

# Use venv python if available
if [ -f "$PROJECT_DIR/.venv/bin/python" ]; then
    PYTHON="$PROJECT_DIR/.venv/bin/python"
else
    PYTHON="python3"
fi

# Run the query
"$PYTHON" "$SCRIPT_DIR/query_gemini_simple.py" \
    --url "$GEMINI_URL" \
    "$@"
