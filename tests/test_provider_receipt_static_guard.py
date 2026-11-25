from __future__ import annotations

import re
from pathlib import Path


DIRECT_PROVIDER_PATTERNS = {
    "openai_chat_completions": re.compile(r"\.chat\.completions\.create\s*\("),
    "openai_responses": re.compile(r"\.responses\.create\s*\("),
    "openai_embeddings": re.compile(r"\.embeddings\.create\s*\("),
    "openai_transcriptions": re.compile(r"\.audio\.transcriptions\.create\s*\("),
    "anthropic_messages": re.compile(r"\.messages\.create\s*\("),
    "gemini_generate_content": re.compile(r"\.models\.generate_content\s*\("),
    "gemini_generate_content_stream": re.compile(
        r"\.models\.generate_content_stream\s*\("
    ),
    "legacy_gemini_generate_content": re.compile(r"\.generate_content\s*\("),
    "litellm_completion": re.compile(r"(?<!litellm_)\bcompletion\s*\("),
}

ALLOWLIST = {
    Path("src/conference_pipeline/provider_receipts.py"),
}


def test_provider_sdk_calls_go_through_receipt_boundary():
    offenders: list[str] = []
    for root in (Path("src"), Path("scripts")):
        for path in sorted(root.rglob("*.py")):
            if path in ALLOWLIST:
                continue
            text = path.read_text(encoding="utf-8")
            for line_no, line in enumerate(text.splitlines(), start=1):
                for name, pattern in DIRECT_PROVIDER_PATTERNS.items():
                    if pattern.search(line):
                        offenders.append(f"{path}:{line_no}: {name}: {line.strip()}")

    assert offenders == []
