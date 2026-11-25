# Gemini Browser Automation

Two ways to automate queries to Google Gemini with your logged-in account.

## Method 1: CDP Connection (Recommended - Uses Your Running Chrome)

This method connects to your already-running Chrome browser, so it automatically uses your existing login session.

### Setup (One Time):

**Step 1: Start Chrome with remote debugging:**
```bash
./scripts/start_chrome_debug.sh
```

This opens Chrome with a debugging port that allows scripts to control it. **You only need to do this once** - keep Chrome running.

**Step 2: Log into Gemini manually:**
- In the Chrome window that opened, navigate to https://gemini.google.com
- Log into your Google account (the /u/2/ one)
- Keep Chrome open

### Usage:

Now you can query Gemini from the command line:

```bash
./scripts/ask_gemini_cdp.sh "What is quantum computing?"
```

**Advantages:**
- ✅ Uses your existing Chrome session (already logged in!)
- ✅ No profile locking issues
- ✅ Can see the browser working in real-time
- ✅ Faster (no new browser launch)

**Disadvantages:**
- Requires Chrome to be running with debug port
- Not headless (you'll see the browser)

## Method 2: Profile Launch (Alternative)

This method launches a new Chrome instance with your profile. **Note: Chrome must be closed for this to work.**

### Setup:

Make sure Chrome is **completely closed**:
```bash
killall "Google Chrome"
```

### Usage:

```bash
./scripts/ask_gemini.sh "What is quantum computing?"
```

**Advantages:**
- ✅ Doesn't require Chrome to be running
- ✅ Can run in headless mode

**Disadvantages:**
- ❌ Won't work if Chrome is already open (profile locked)
- ❌ May not preserve login session properly
- ❌ Slower (launches new browser)

## Recommended Workflow

**For interactive use while Chrome is running:**
```bash
# Start Chrome with debugging (one time)
./scripts/start_chrome_debug.sh

# Log into Gemini manually in that Chrome window

# Now query from command line
./scripts/ask_gemini_cdp.sh "Explain neural networks"
./scripts/ask_gemini_cdp.sh "What are the key topics at NeurIPS 2025?"
```

**For automated/headless use:**
```bash
# Close Chrome
killall "Google Chrome"

# Run headless
.venv/bin/python scripts/query_gemini_browser.py "Question here" \
    --auto-profile \
    --url "https://gemini.google.com/u/2/app" \
    --headless
```

## Python API

Both methods can be used programmatically:

### CDP Method (for running Chrome):
```python
from scripts.query_gemini_simple import query_gemini_via_cdp

response = query_gemini_via_cdp(
    question="What is machine learning?",
    gemini_url="https://gemini.google.com/u/2/app?pageId=none"
)
print(response)
```

### Profile Method (Chrome must be closed):
```python
from scripts.query_gemini_browser import query_gemini

response = query_gemini(
    question="What is machine learning?",
    chrome_profile_dir=os.path.expanduser("~/Library/Application Support/Google/Chrome/Default"),
    gemini_url="https://gemini.google.com/u/2/app?pageId=none"
)
print(response)
```

## Troubleshooting

### "Could not connect to Chrome" (CDP method)
- Make sure Chrome is running with remote debugging: `./scripts/start_chrome_debug.sh`
- Check if port 9222 is open: `lsof -i :9222`

### "Profile directory is locked" (Profile method)
- Close all Chrome instances: `killall "Google Chrome"`
- Wait a few seconds and try again

### "Not logged in"
- For CDP method: Manually log into Gemini in the Chrome window
- For Profile method: Try the CDP method instead

### Response extraction issues
- Screenshots are saved to `gemini_debug.png` or `gemini_response.png`
- Check these to see what the page looked like

## Files

- `scripts/query_gemini_simple.py` - CDP connection method
- `scripts/query_gemini_browser.py` - Profile launch method
- `scripts/ask_gemini_cdp.sh` - Convenience wrapper for CDP
- `scripts/ask_gemini.sh` - Convenience wrapper for profile
- `scripts/start_chrome_debug.sh` - Helper to start Chrome with debugging
