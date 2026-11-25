#!/bin/bash
# Convenience script to query Gemini using your logged-in Chrome profile

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Your Chrome profile location (adjust if different)
PROFILE_DIR="$HOME/Library/Application Support/Google/Chrome/Default"

# Your Gemini URL with account selector
GEMINI_URL="https://gemini.google.com/u/2/app?pageId=none"

# Use venv python if available, otherwise system python3
if [ -f "$PROJECT_DIR/.venv/bin/python" ]; then
    PYTHON="$PROJECT_DIR/.venv/bin/python"
else
    PYTHON="python3"
fi

# Run the query
"$PYTHON" "$SCRIPT_DIR/query_gemini_browser.py" \
    --profile "$PROFILE_DIR" \
    --url "$GEMINI_URL" \
    "$@"
