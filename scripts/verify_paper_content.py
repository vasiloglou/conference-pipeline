#!/usr/bin/env python3
"""Verify paper content against expected titles using Gemini."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import fitz  # PyMuPDF
from dotenv import load_dotenv

from conference_pipeline.model_governance import default_model_for_provider
from conference_pipeline.provider_receipts import (
    gemini_generate_content,
    infer_conference_workspace,
)

load_dotenv()

try:
    from google import genai
except ImportError as exc:
    raise SystemExit("Missing: pip install google-genai") from exc


def extract_pdf_text(pdf_path: Path, max_pages: int = 3) -> str:
    """Extract text from first few pages of a PDF."""
    try:
        doc = fitz.open(pdf_path)
        text_parts = []
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            text_parts.append(page.get_text())
        doc.close()
        return "\n".join(text_parts)[:8000]  # Limit to ~8k chars
    except Exception as e:
        return f"Error reading PDF: {e}"


def verify_with_gemini(
    expected_title: str,
    pdf_text: str,
    model_name: str = default_model_for_provider("qa.paper_content_verification", "gemini"),
) -> dict:
    """Use Gemini to verify if PDF content matches expected paper."""

    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    )

    prompt = f"""You are verifying if a downloaded PDF contains the expected research paper.

EXPECTED PAPER TITLE: {expected_title}

PDF CONTENT (first pages):
{pdf_text[:6000]}

Analyze whether this PDF is the correct paper. Consider:
1. Does the paper title in the PDF match or closely relate to the expected title?
2. Look for key identifying words/acronyms (e.g., "ProRL", "TabNet", "GRIFFIN")
3. Minor title variations are OK (e.g., "XYZ: A Method" vs "XYZ: Novel Method")
4. If it's the wrong paper, what paper does this PDF actually contain?

Return a JSON object with:
{{
  "is_correct": true/false,
  "confidence": "high"/"medium"/"low",
  "actual_title": "the actual paper title if you can determine it",
  "reason": "brief explanation",
  "key_mismatch": "what key term/concept is missing or wrong" (if incorrect)
}}

Output ONLY valid JSON."""

    try:
        response = gemini_generate_content(client, caller=__file__,
            model=model_name,
            contents=prompt,
            config={"temperature": 0.1, "max_output_tokens": 1000}
        )

        text = response.text.strip()
        # Clean up markdown
        if "```" in text:
            import re
            match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
            if match:
                text = match.group(1).strip()

        return json.loads(text)
    except Exception as e:
        return {"error": str(e), "is_correct": None}


def verify_papers(
    report_path: Path,
    output_path: Path,
    limit: int = None,
    status_filter: str = "bad"
) -> dict:
    """Verify papers from sanity report and create corrected report."""

    with open(report_path) as f:
        data = json.load(f)

    records = [r for r in data["records"] if r["status"] == status_filter]
    if limit:
        records = records[:limit]

    print(f"Verifying {len(records)} papers with status '{status_filter}'...", flush=True)

    results = {
        "actually_correct": [],
        "needs_redownload": [],
        "verification_failed": [],
        "details": []
    }

    for i, record in enumerate(records):
        print(f"[{i+1}/{len(records)}] Checking: {record['expected_title'][:50]}...", flush=True)

        pdf_path = Path(record["file_path"])
        if not pdf_path.exists():
            print(f"  -> PDF not found: {pdf_path}")
            results["verification_failed"].append({
                **record,
                "verification": {"error": "PDF not found"}
            })
            continue

        # Extract text
        pdf_text = extract_pdf_text(pdf_path)
        if pdf_text.startswith("Error"):
            print(f"  -> {pdf_text}")
            results["verification_failed"].append({
                **record,
                "verification": {"error": pdf_text}
            })
            continue

        # Verify with Gemini
        verification = verify_with_gemini(record["expected_title"], pdf_text)

        if verification.get("is_correct"):
            print(f"  -> CORRECT ({verification.get('confidence', 'unknown')} confidence)", flush=True)
            results["actually_correct"].append({
                **record,
                "verification": verification,
                "new_status": "ok"
            })
        elif verification.get("is_correct") is False:
            print(f"  -> WRONG: {verification.get('reason', 'unknown')[:50]}", flush=True)
            results["needs_redownload"].append({
                **record,
                "verification": verification,
                "actual_paper": verification.get("actual_title", "unknown")
            })
        else:
            print(f"  -> UNCERTAIN: {verification.get('error', 'unknown')}", flush=True)
            results["verification_failed"].append({
                **record,
                "verification": verification
            })

        results["details"].append({**record, "verification": verification})

        # Rate limiting
        time.sleep(0.5)

    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Actually correct: {len(results['actually_correct'])}")
    print(f"Needs re-download: {len(results['needs_redownload'])}")
    print(f"Verification failed: {len(results['verification_failed'])}")

    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Verify paper content")
    parser.add_argument(
        "report_path",
        type=Path,
        help="Path to title_sanity_report.json"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("verification_results.json"),
        help="Output path for verification results"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        help="Limit number of papers to verify"
    )
    parser.add_argument(
        "--status",
        default="bad",
        help="Filter by status (bad, warn, ok)"
    )
    args = parser.parse_args()

    # D-QUICK-260715-rz1: this script has no --workspace flag, so derive one
    # from the report_path positional arg to avoid falling back to
    # Path.cwd() (repo root).
    inferred_workspace = infer_conference_workspace(args.report_path)
    if inferred_workspace is not None:
        os.environ.setdefault("PROVIDER_RECEIPT_WORKSPACE", str(inferred_workspace))
    os.environ.setdefault("PROVIDER_RECEIPT_STAGE_ID", "verify_paper_content")
    os.environ.setdefault("PROVIDER_RECEIPT_CALLER", __file__)

    verify_papers(args.report_path, args.output, args.limit, args.status)


if __name__ == "__main__":
    main()
