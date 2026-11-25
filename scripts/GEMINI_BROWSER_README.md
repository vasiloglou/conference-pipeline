# Gemini Browser Automation

Query Google Gemini via browser automation using Playwright.

## Features

- ✅ Opens Chrome browser and navigates to gemini.google.com
- ✅ Handles Google authentication (prompts you to log in if needed)
- ✅ Submits your question
- ✅ Waits for response to complete
- ✅ Extracts and returns the answer
- ✅ Works with or without visible browser (headless mode)

## Prerequisites

Playwright should already be installed (it's in requirements.txt). If not:

```bash
pip install playwright
playwright install chromium chrome
```

## Usage

### Basic Usage

```bash
python3 scripts/query_gemini_browser.py "What is the capital of France?"
```

### Interactive Mode

```bash
python3 scripts/query_gemini_browser.py
# You'll be prompted to enter your question
```

### Headless Mode (No Visible Browser)

```bash
python3 scripts/query_gemini_browser.py "Explain quantum computing" --headless
```

### As a Python Module

```python
from scripts.query_gemini_browser import query_gemini

response = query_gemini(
    question="What are the main topics at NeurIPS 2025?",
    headless=False,
    timeout=60000
)

print(response)
```

## How It Works

1. **Opens Browser**: Launches Chrome using Playwright
2. **Navigates**: Goes to gemini.google.com
3. **Login Detection**: If you're not logged in, it waits for you to authenticate
4. **Input Finding**: Locates the chat input using multiple selector strategies
5. **Question Submission**: Types your question and clicks send (or presses Enter)
6. **Response Waiting**: Waits for response to complete by detecting text stabilization
7. **Extraction**: Extracts the response text using multiple fallback strategies
8. **Return**: Returns the clean response text

## Authentication

- **First Run**: The browser will open and you'll need to log in to your Google account
- **Subsequent Runs**: If you stay logged in, it should work automatically
- **Timeout**: You have 2 minutes to complete login

## Troubleshooting

### "Could not find chat input"
- The page selectors may have changed
- A screenshot `gemini_debug.png` will be saved for inspection
- Try running in non-headless mode to see what's happening

### "Login timeout"
- Log in faster (you have 2 minutes)
- Try running without `--headless` flag

### Response extraction issues
- A screenshot `gemini_response.png` will be saved
- The script tries multiple strategies to find the response
- Check the screenshot to see what the page looks like

## Options

```bash
python3 scripts/query_gemini_browser.py --help

positional arguments:
  question              Question to ask Gemini

optional arguments:
  --headless            Run in headless mode (no visible browser)
  --timeout TIMEOUT     Navigation timeout in milliseconds (default: 60000)
```

## Example Output

```bash
$ python3 scripts/query_gemini_browser.py "What is 2+2?"

Opening Chrome and navigating to Gemini...
Question: What is 2+2?

📍 Navigating to gemini.google.com...
🔍 Looking for chat input...
✓ Found input: div[contenteditable="true"]
⌨️  Typing question...
🔍 Looking for send button...
✓ Found send button: button[aria-label*="Send"]
⏳ Waiting for response...
✓ Found response container: [data-message-author="model"]
⏳ Waiting for response to complete...
   Response length: 156 chars...
✓ Response complete!
📝 Extracting response...

✓ Done! Closing browser in 3 seconds...

================================================================================
GEMINI RESPONSE:
================================================================================
2 + 2 = 4
================================================================================
```

## Notes

- This uses browser automation, so it's slower than using the Gemini API directly
- Useful when you need to test Gemini's web interface or don't have API access
- The script tries to be robust with multiple fallback strategies for finding elements
- Response extraction waits for text to stabilize (no changes for 3 seconds)
