#!/usr/bin/env python3
"""Generate focused podcasts using ranked facts.

This script reads a fact ranking file and generates a podcast that prioritizes
high-importance facts, ensuring the most critical content is covered.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from conference_pipeline.provider_receipts import (
    anthropic_messages_create,
    gemini_generate_content,
    infer_conference_workspace,
)
from conference_pipeline.model_governance import default_model, default_provider, validate_selection


MODEL_TASK_ID = "podcast.focused_script_generation"
DEFAULT_PROVIDER = default_provider(MODEL_TASK_ID)
DEFAULT_MODEL = default_model(MODEL_TASK_ID)
WORDS_PER_MINUTE = 130


@dataclass
class ScriptConfig:
    title: str
    host_a: str
    host_b: str
    two_hosts: bool
    target_minutes: int
    max_segment_chars: int
    temperature: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a focused podcast from ranked facts."
    )
    parser.add_argument(
        "input",
        help="Path to the source document (markdown).",
    )
    parser.add_argument(
        "--fact-ranks",
        required=True,
        help="Path to the fact ranking JSON file (from rank_facts.py).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for the podcast.",
    )
    parser.add_argument(
        "--target-minutes",
        type=int,
        default=20,
        help="Target podcast duration in minutes (default: 20).",
    )
    parser.add_argument(
        "--top-n-facts",
        type=int,
        default=50,
        help="Number of top-ranked facts to include (default: 50).",
    )
    parser.add_argument(
        "--max-tier",
        type=int,
        default=3,
        choices=[1, 2, 3, 4],
        help="Maximum tier to include (1=essential only, 4=all, default: 3).",
    )
    parser.add_argument(
        "--provider",
        choices=("anthropic", "gemini"),
        default=DEFAULT_PROVIDER,
        help=f"Provider to use (default: {DEFAULT_PROVIDER}).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Governed model to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--two-hosts",
        action="store_true",
        help="Use two hosts instead of one.",
    )
    parser.add_argument(
        "--host-a",
        default="Host A",
        help="Name of the first host.",
    )
    parser.add_argument(
        "--host-b",
        default="Host B",
        help="Name of the second host (if --two-hosts).",
    )
    parser.add_argument(
        "--title",
        default="",
        help="Podcast title (auto-generated if not provided).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Sampling temperature (default: 0.4).",
    )
    parser.add_argument(
        "--max-segment-chars",
        type=int,
        default=1000,
        help="Maximum characters per segment (default: 1000).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()

    # D-QUICK-260715-rz1: this script has no --workspace flag, so derive one
    # from --output-dir to avoid falling back to Path.cwd() (repo root). Note
    # that when this script is invoked as a subprocess by run_icml2026/iclr2026
    # _production.py, PROVIDER_RECEIPT_WORKSPACE is already inherited from the
    # orchestrator's provider_receipt_context (os.environ.setdefault below is
    # then a no-op).
    inferred_workspace = infer_conference_workspace(args.output_dir)
    if inferred_workspace is not None:
        os.environ.setdefault("PROVIDER_RECEIPT_WORKSPACE", str(inferred_workspace))
    os.environ.setdefault("PROVIDER_RECEIPT_STAGE_ID", "generate_focused_podcast")
    os.environ.setdefault("PROVIDER_RECEIPT_CALLER", __file__)

    source_path = Path(args.input)
    if not source_path.exists():
        raise SystemExit(f"Source file not found: {source_path}")

    ranks_path = Path(args.fact_ranks)
    if not ranks_path.exists():
        raise SystemExit(f"Fact ranks file not found: {ranks_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    validate_selection(
        MODEL_TASK_ID,
        provider=args.provider,
        model=args.model,
    )
    if args.provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise SystemExit("ANTHROPIC_API_KEY not set.")
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise SystemExit("Missing dependency: anthropic.") from exc
        client = Anthropic(api_key=api_key, timeout=300)
    else:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit("GOOGLE_API_KEY or GEMINI_API_KEY not set.")
        try:
            from google import genai
        except ImportError as exc:
            raise SystemExit(
                "Missing dependency: google-genai. Install with `pip install google-genai`."
            ) from exc
        client = genai.Client(api_key=api_key)

    # Load source and rankings
    source_text = source_path.read_text()
    ranks_data = json.loads(ranks_path.read_text())

    # Filter facts by tier and top N
    selected_facts = select_facts(
        ranks_data,
        max_tier=args.max_tier,
        top_n=args.top_n_facts,
    )

    print(f"Source: {source_path}")
    print(f"Rankings: {ranks_path}")
    print(f"Selected {len(selected_facts)} facts (tier <= {args.max_tier}, top {args.top_n_facts})")
    print(f"Target: {args.target_minutes} minutes")
    print()

    # Build config
    config = ScriptConfig(
        title=args.title or f"Focused Podcast: {source_path.stem}",
        host_a=args.host_a,
        host_b=args.host_b,
        two_hosts=args.two_hosts,
        target_minutes=args.target_minutes,
        max_segment_chars=args.max_segment_chars,
        temperature=args.temperature,
    )

    # Generate script
    print("Generating focused podcast script...")
    script = generate_focused_script(
        client=client,
        provider=args.provider,
        model=args.model,
        source_text=source_text,
        selected_facts=selected_facts,
        config=config,
    )

    if not script:
        raise SystemExit("Failed to generate script.")

    # Write outputs
    write_outputs(output_dir, script, source_path, selected_facts)

    word_count = count_script_words(script)
    print(f"\nGenerated script: {word_count} words")
    print(f"Results written to {output_dir}")


def select_facts(
    ranks_data: dict,
    max_tier: int,
    top_n: int,
) -> list[dict]:
    """Select facts based on tier and ranking."""
    facts = ranks_data.get("facts", [])

    # Filter by tier
    filtered = [f for f in facts if f.get("tier", 4) <= max_tier]

    # Take top N
    return filtered[:top_n]


def compute_script_max_tokens(target_minutes: int) -> int:
    """Token budget for the focused-podcast script-generation call.

    Must comfortably fit one JSON object covering up to ``--top-n-facts``
    facts in detail across alternating two-host segments; too low a ceiling
    truncates the JSON response before it can close. The prior formula
    (``min(16000, max(4096, target_words * 2))``) clamped to exactly 4096
    tokens for any ``target_minutes`` below ~15.75 -- including this
    platform's locked 12-minute podcast convention -- and was observed to
    truncate every one of 5 real live responses at exactly that ceiling
    (CVPR2026 Phase 9 Plan 09-04, 2026-07-28; see
    ``.planning/workstreams/cvpr-2026/phases/09-rendered-podcast-production/deferred-items.md``).
    """
    target_words = target_minutes * WORDS_PER_MINUTE
    return min(16000, max(8192, target_words * 4))


def generate_focused_script(
    *,
    client,
    provider: str,
    model: str,
    source_text: str,
    selected_facts: list[dict],
    config: ScriptConfig,
    max_retries: int = 5,
) -> dict | None:
    """Generate a podcast script focused on selected high-priority facts."""
    target_words = config.target_minutes * WORDS_PER_MINUTE
    min_words = int(target_words * 0.9)

    # Build the facts list for the prompt
    facts_text = "\n".join(
        f"- [Tier {f['tier']}, Score {f['score']:.2f}] {f['text']}"
        for f in selected_facts[:30]  # Limit to top 30 for prompt size
    )

    if config.two_hosts:
        host_line = f"Hosts: {config.host_a} and {config.host_b}."
        speaker_rule = "Use both hosts and alternate every 1-2 segments."
    else:
        host_line = f"Single host: {config.host_a}."
        speaker_rule = f"Use only {config.host_a} in every segment."

    prompt = f"""Create a podcast script that MUST cover the following HIGH-PRIORITY FACTS.

{host_line}

CRITICAL REQUIREMENTS:
1. The script MUST contain at least {min_words} words (target: {target_words} words).
2. You MUST cover ALL of the high-priority facts listed below.
3. Each fact should be discussed in detail with context and explanation.
4. Facts with higher tiers (Tier 1 > Tier 2 > Tier 3) should get more discussion time.

Target duration: {config.target_minutes} minutes.
Max segment length: {config.max_segment_chars} characters.
Tone: engaging, informative, and thorough.

Rules:
- Do not add facts not present in the source or fact list.
- If quoting from the source, quote exactly.
- Each segment should be 2-5 sentences.
- Do not claim or imply that the source material's "Source Anchors" section (or its topic-report list, extraction ids, or research ids) enumerates every topic by number. If the source separately states an overall coverage or traceability status (for example, "all N topics well-covered" or "traceability status pass across all N topics"), you may cite that status accurately, but do not conflate it with the specific anchors/citations actually listed -- the anchors list may cover only a subset of the topics referenced elsewhere.
- {speaker_rule}
- Output JSON only.

HIGH-PRIORITY FACTS TO COVER:
{facts_text}

FULL SOURCE (for context and additional details):
{source_text[:40000]}

JSON schema:
{{"title": "string", "summary": "string", "segments": [{{"speaker": "string", "text": "string"}}], "show_notes": "string"}}
"""

    last_error = None
    for attempt in range(max_retries):
        try:
            if provider == "anthropic":
                response = anthropic_messages_create(
                    client,
                    caller=__file__,
                    model=model,
                    max_tokens=compute_script_max_tokens(config.target_minutes),
                    system="Return one JSON object only, with no markdown fences.",
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = "".join(
                    str(block.get("text") or "")
                    if isinstance(block, dict)
                    else str(getattr(block, "text", "") or "")
                    for block in (getattr(response, "content", None) or [])
                )
            elif provider == "gemini":
                response = gemini_generate_content(
                    client,
                    caller=__file__,
                    model=model,
                    contents=prompt,
                    config={
                        "temperature": config.temperature,
                        "response_mime_type": "application/json",
                    },
                )
                raw = response.text or ""
            else:
                raise ValueError(f"Unsupported podcast script provider: {provider}")
            if not raw:
                print(f"  Attempt {attempt + 1}: empty response", file=sys.stderr)
                time.sleep(2)
                continue

            parsed = json.loads(raw)

            # Handle malformed response (list instead of dict)
            if isinstance(parsed, list):
                print(f"  Attempt {attempt + 1}: Got list instead of dict, wrapping...", file=sys.stderr)
                # Normalize list items to segment format
                normalized_segments = []
                for item in parsed:
                    if isinstance(item, str):
                        normalized_segments.append({"speaker": config.host_a, "text": item})
                    elif isinstance(item, dict):
                        if "text" not in item and "content" in item:
                            item["text"] = item.pop("content")
                        if "speaker" not in item:
                            item["speaker"] = config.host_a
                        normalized_segments.append(item)
                script = {
                    "title": config.title,
                    "summary": "",
                    "segments": normalized_segments,
                    "show_notes": "",
                }
            else:
                script = parsed

            # Ensure required fields
            if "title" not in script:
                script["title"] = config.title
            if "summary" not in script:
                script["summary"] = ""
            if "segments" not in script:
                script["segments"] = []
            if "show_notes" not in script:
                script["show_notes"] = ""

            # Check word count - retry if too low
            word_count = sum(
                len(str(seg.get("text", "")).split())
                for seg in script.get("segments", [])
                if isinstance(seg, dict)
            )
            if word_count < 100:
                print(f"  Attempt {attempt + 1}: Only {word_count} words, retrying...", file=sys.stderr)
                time.sleep(2)
                continue

            return script

        except json.JSONDecodeError as e:
            last_error = e
            print(f"  Attempt {attempt + 1}: JSON parse error - {e}", file=sys.stderr)
            time.sleep(2)
        except Exception as e:
            last_error = e
            print(f"  Attempt {attempt + 1}: {e}", file=sys.stderr)
            time.sleep(5)

    print(f"Failed after {max_retries} attempts. Last error: {last_error}", file=sys.stderr)
    return None


def count_script_words(script: dict) -> int:
    """Count total words in all script segments."""
    total = 0
    for seg in script.get("segments", []):
        text = str(seg.get("text", "")).strip()
        total += len(text.split())
    return total


def write_outputs(
    output_dir: Path,
    script: dict,
    source_path: Path,
    selected_facts: list[dict],
) -> None:
    """Write script and metadata outputs."""
    # Script JSON
    (output_dir / "script.json").write_text(
        json.dumps(script, indent=2), encoding="utf-8"
    )

    # Script Markdown
    md_lines = [
        f"# {script.get('title', 'Podcast')}",
        "",
        "## Summary",
        script.get("summary", ""),
        "",
        "## Script",
        "",
    ]

    for seg in script.get("segments", []):
        speaker = seg.get("speaker", "Host")
        text = seg.get("text", "")
        md_lines.append(f"**{speaker}:** {text}")
        md_lines.append("")

    (output_dir / "script.md").write_text("\n".join(md_lines), encoding="utf-8")

    # Show notes
    show_notes = script.get("show_notes", "")
    notes_lines = [
        f"# Show Notes: {script.get('title', 'Podcast')}",
        "",
        f"**Source:** {source_path.name}",
        f"**Facts covered:** {len(selected_facts)}",
        "",
        "## Topics",
        "",
        show_notes if show_notes else "See script for details.",
        "",
        "## Fact Coverage",
        "",
        "### Tier 1 (Essential)",
        "",
    ]

    for tier in range(1, 5):
        tier_facts = [f for f in selected_facts if f.get("tier") == tier]
        tier_names = {1: "Essential", 2: "Important", 3: "Supporting", 4: "Comprehensive"}

        if tier > 1:
            notes_lines.extend([
                "",
                f"### Tier {tier} ({tier_names[tier]})",
                "",
            ])

        for f in tier_facts[:5]:
            preview = f["text"][:100] + "..." if len(f["text"]) > 100 else f["text"]
            notes_lines.append(f"- {preview}")

        if len(tier_facts) > 5:
            notes_lines.append(f"- *...and {len(tier_facts) - 5} more*")

    (output_dir / "show_notes.md").write_text("\n".join(notes_lines), encoding="utf-8")

    # Metadata
    metadata = {
        "source": str(source_path),
        "facts_selected": len(selected_facts),
        "word_count": count_script_words(script),
        "segments": len(script.get("segments", [])),
        "tier_distribution": {
            f"tier_{tier}": sum(1 for f in selected_facts if f.get("tier") == tier)
            for tier in range(1, 5)
        },
    }

    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
