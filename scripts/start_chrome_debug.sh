#!/bin/bash
# Start Chrome with remote debugging enabled for automation

echo "Starting Chrome with remote debugging on port 9222..."
echo "This allows scripts to control your Chrome browser."
echo ""

# Use a dedicated user data dir to avoid locking your main profile
DEBUG_PROFILE_DIR="$HOME/.chrome-remote-debug"
mkdir -p "$DEBUG_PROFILE_DIR"

# Kill existing Chrome instances (optional - comment out if you don't want this)
# killall "Google Chrome" 2>/dev/null

# Start Chrome with debugging port using the dedicated profile
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
  --remote-debugging-port=9222 \
  --user-data-dir="$DEBUG_PROFILE_DIR" \
  > /dev/null 2>&1 &

echo "✓ Chrome started with remote debugging (profile: $DEBUG_PROFILE_DIR)"
echo ""
echo "Now you can run:"
echo "  ./scripts/ask_gemini_cdp.sh \"Your question here\""
