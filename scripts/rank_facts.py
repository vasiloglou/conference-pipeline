#!/usr/bin/env python3
"""Rank facts from source documents using telescopic and frequency analysis.

This script implements fact importance ranking using two approaches:
1. Telescopic: Generate podcasts at multiple durations and rank facts by inclusion tier
2. Frequency: Run multiple short podcast generations with scrambled input and count frequency

The output is a ranked list of facts that can be used for focused podcast generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from conference_pipeline.model_governance.registry import default_model_for_provider
from conference_pipeline.provider_receipts import (
    gemini_generate_content,
    infer_conference_workspace,
)

try:
    from google import genai
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: google-genai. Install with `pip install google-genai`."
    ) from exc


DEFAULT_MODEL = default_model_for_provider("ranking.fact_importance", "gemini")
WORDS_PER_MINUTE = 130


@dataclass
class Fact:
    """Represents an extracted fact from a podcast script."""

    text: str
    hash: str = field(default="")

    def __post_init__(self):
        if not self.hash:
            # Create a normalized hash for comparison
            normalized = self.normalize(self.text)
            self.hash = hashlib.md5(normalized.encode()).hexdigest()[:12]

    @staticmethod
    def normalize(text: str) -> str:
        """Normalize text for comparison."""
        # Lowercase, remove extra whitespace, remove punctuation
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s]", "", text)
        return text.strip()


@dataclass
class RankedFact:
    """A fact with its importance score and metadata."""

    fact: Fact
    score: float
    tier: int  # 1 = essential, 2 = important, 3 = supporting, 4 = comprehensive
    frequency: float  # 0-1, how often it appears in frequency analysis
    sources: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank facts from source documents by importance."
    )
    parser.add_argument(
        "input",
        help="Path to the source document (markdown).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for ranking results.",
    )
    parser.add_argument(
        "--method",
        choices=["telescopic", "frequency", "hybrid"],
        default="hybrid",
        help="Ranking method (default: hybrid).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Gemini model to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=5,
        help="Number of runs for frequency analysis (default: 5).",
    )
    parser.add_argument(
        "--durations",
        type=str,
        default="10,20,30,40",
        help="Comma-separated durations for telescopic analysis (default: 10,20,30,40).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation (default: 0.7).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()

    random.seed(args.seed)

    source_path = Path(args.input)
    if not source_path.exists():
        raise SystemExit(f"Source file not found: {source_path}")

    # D-QUICK-260715-rz1: this script has no --workspace flag, so derive one
    # from the source path to avoid falling back to Path.cwd() (repo root).
    inferred_workspace = infer_conference_workspace(source_path)
    if inferred_workspace is not None:
        os.environ.setdefault("PROVIDER_RECEIPT_WORKSPACE", str(inferred_workspace))
    os.environ.setdefault("PROVIDER_RECEIPT_STAGE_ID", "rank_facts")
    os.environ.setdefault("PROVIDER_RECEIPT_CALLER", __file__)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY not set.")

    client = genai.Client(api_key=api_key)
    source_text = source_path.read_text()

    print(f"Source: {source_path}")
    print(f"Method: {args.method}")
    print(f"Output: {output_dir}")
    print()

    # Parse durations
    durations = [int(d.strip()) for d in args.durations.split(",")]

    if args.method == "telescopic":
        ranked_facts = rank_telescopic(
            client=client,
            model=args.model,
            source_text=source_text,
            durations=durations,
            temperature=args.temperature,
            output_dir=output_dir,
        )
    elif args.method == "frequency":
        ranked_facts = rank_frequency(
            client=client,
            model=args.model,
            source_text=source_text,
            n_runs=args.n_runs,
            temperature=args.temperature,
            output_dir=output_dir,
        )
    else:  # hybrid
        ranked_facts = rank_hybrid(
            client=client,
            model=args.model,
            source_text=source_text,
            durations=durations,
            n_runs=args.n_runs,
            temperature=args.temperature,
            output_dir=output_dir,
        )

    # Write outputs
    write_outputs(ranked_facts, output_dir)

    print(f"\nRanked {len(ranked_facts)} facts")
    print(f"Results written to {output_dir}")


def rank_telescopic(
    *,
    client,
    model: str,
    source_text: str,
    durations: list[int],
    temperature: float,
    output_dir: Path,
) -> list[RankedFact]:
    """Rank facts using telescopic generation at multiple durations."""
    print("=== Telescopic Analysis ===")

    # Generate scripts at each duration
    duration_facts: dict[int, set[str]] = {}

    for duration in sorted(durations):
        print(f"\nGenerating {duration}-minute script...")
        script = generate_script(
            client=client,
            model=model,
            source_text=source_text,
            target_minutes=duration,
            temperature=temperature,
        )

        if script:
            # Save script
            script_dir = output_dir / "telescopic_scripts"
            script_dir.mkdir(exist_ok=True)
            (script_dir / f"script_{duration}min.json").write_text(
                json.dumps(script, indent=2)
            )

            # Extract facts
            facts = extract_facts_from_script(script)
            duration_facts[duration] = {f.hash for f in facts}
            print(f"  Extracted {len(facts)} facts")
        else:
            print(f"  Failed to generate script", file=sys.stderr)
            duration_facts[duration] = set()

    # Rank facts by tier (which duration they first appear in)
    all_facts: dict[str, RankedFact] = {}
    sorted_durations = sorted(durations)

    for duration in sorted_durations:
        for script_dir in [output_dir / "telescopic_scripts"]:
            script_path = script_dir / f"script_{duration}min.json"
            if script_path.exists():
                script = json.loads(script_path.read_text())
                facts = extract_facts_from_script(script)

                for fact in facts:
                    if fact.hash not in all_facts:
                        # Determine tier based on first appearance
                        tier = sorted_durations.index(duration) + 1
                        # Score decreases with tier (tier 1 = 1.0, tier 4 = 0.25)
                        score = 1.0 / tier
                        all_facts[fact.hash] = RankedFact(
                            fact=fact,
                            score=score,
                            tier=tier,
                            frequency=1.0,
                            sources=[f"{duration}min"],
                        )
                    else:
                        # Add source
                        all_facts[fact.hash].sources.append(f"{duration}min")

    # Sort by score (descending)
    return sorted(all_facts.values(), key=lambda x: (-x.score, x.tier))


def rank_frequency(
    *,
    client,
    model: str,
    source_text: str,
    n_runs: int,
    temperature: float,
    output_dir: Path,
    target_minutes: int = 10,
) -> list[RankedFact]:
    """Rank facts using frequency analysis with scrambled input."""
    print("=== Frequency Analysis ===")

    fact_counts: Counter[str] = Counter()
    fact_objects: dict[str, Fact] = {}

    runs_dir = output_dir / "frequency_runs"
    runs_dir.mkdir(exist_ok=True)

    for run in range(1, n_runs + 1):
        print(f"\nRun {run}/{n_runs}...")

        # Scramble sections
        scrambled = scramble_sections(source_text)

        # Generate script
        script = generate_script(
            client=client,
            model=model,
            source_text=scrambled,
            target_minutes=target_minutes,
            temperature=temperature,
        )

        if script:
            # Save script
            (runs_dir / f"run_{run:02d}.json").write_text(
                json.dumps(script, indent=2)
            )

            # Extract and count facts
            facts = extract_facts_from_script(script)
            for fact in facts:
                fact_counts[fact.hash] += 1
                if fact.hash not in fact_objects:
                    fact_objects[fact.hash] = fact

            print(f"  Extracted {len(facts)} facts")
        else:
            print(f"  Failed to generate script", file=sys.stderr)

        # Rate limiting
        time.sleep(1)

    # Calculate frequency scores
    ranked_facts = []
    for fact_hash, count in fact_counts.most_common():
        frequency = count / n_runs
        # Score is simply the frequency
        score = frequency

        # Determine tier based on frequency
        if frequency >= 0.8:
            tier = 1  # Essential (80%+ appearance)
        elif frequency >= 0.6:
            tier = 2  # Important (60-80%)
        elif frequency >= 0.4:
            tier = 3  # Supporting (40-60%)
        else:
            tier = 4  # Comprehensive (<40%)

        ranked_facts.append(
            RankedFact(
                fact=fact_objects[fact_hash],
                score=score,
                tier=tier,
                frequency=frequency,
                sources=[f"run_{i}" for i in range(1, count + 1)],
            )
        )

    return sorted(ranked_facts, key=lambda x: -x.score)


def rank_hybrid(
    *,
    client,
    model: str,
    source_text: str,
    durations: list[int],
    n_runs: int,
    temperature: float,
    output_dir: Path,
) -> list[RankedFact]:
    """Combine telescopic and frequency analysis for robust ranking."""
    print("=== Hybrid Analysis ===")

    # Phase 1: Frequency analysis
    print("\n--- Phase 1: Frequency Analysis ---")
    freq_facts = rank_frequency(
        client=client,
        model=model,
        source_text=source_text,
        n_runs=n_runs,
        temperature=temperature,
        output_dir=output_dir,
    )
    freq_scores = {rf.fact.hash: rf.frequency for rf in freq_facts}

    # Phase 2: Telescopic analysis
    print("\n--- Phase 2: Telescopic Analysis ---")
    tele_facts = rank_telescopic(
        client=client,
        model=model,
        source_text=source_text,
        durations=durations,
        temperature=temperature,
        output_dir=output_dir,
    )
    tele_scores = {rf.fact.hash: rf.score for rf in tele_facts}

    # Combine scores: 0.6 * frequency + 0.4 * tier_score
    combined: dict[str, RankedFact] = {}

    # Start with frequency facts
    for rf in freq_facts:
        freq_score = rf.frequency
        tele_score = tele_scores.get(rf.fact.hash, 0)
        combined_score = 0.6 * freq_score + 0.4 * tele_score

        combined[rf.fact.hash] = RankedFact(
            fact=rf.fact,
            score=combined_score,
            tier=rf.tier,
            frequency=rf.frequency,
            sources=rf.sources,
        )

    # Add telescopic-only facts
    for rf in tele_facts:
        if rf.fact.hash not in combined:
            freq_score = freq_scores.get(rf.fact.hash, 0)
            combined_score = 0.6 * freq_score + 0.4 * rf.score

            combined[rf.fact.hash] = RankedFact(
                fact=rf.fact,
                score=combined_score,
                tier=rf.tier,
                frequency=0,
                sources=rf.sources,
            )

    # Re-calculate tiers based on combined score
    sorted_facts = sorted(combined.values(), key=lambda x: -x.score)
    total = len(sorted_facts)

    for i, rf in enumerate(sorted_facts):
        percentile = i / total if total > 0 else 0
        if percentile < 0.25:
            rf.tier = 1
        elif percentile < 0.50:
            rf.tier = 2
        elif percentile < 0.75:
            rf.tier = 3
        else:
            rf.tier = 4

    return sorted_facts


def scramble_sections(source_text: str) -> str:
    """Scramble major sections while preserving internal structure."""
    # Split by level 2 headers (##)
    sections = re.split(r"(?=^## )", source_text, flags=re.MULTILINE)

    # Keep the preamble (before first ##) in place
    preamble = ""
    if sections and not sections[0].startswith("##"):
        preamble = sections[0]
        sections = sections[1:]

    # Shuffle the sections
    random.shuffle(sections)

    # Reconstruct
    return preamble + "".join(sections)


def generate_script(
    *,
    client,
    model: str,
    source_text: str,
    target_minutes: int,
    temperature: float,
) -> dict | None:
    """Generate a podcast script from source text."""
    target_words = target_minutes * WORDS_PER_MINUTE

    prompt = f"""Create a podcast script based ONLY on the source below.

Single host podcast. Target length: {target_words} words ({target_minutes} minutes).

Rules:
- Do not add facts not present in the source.
- Cover the most important points from the source material.
- Include specific examples, statistics, and details.
- Output JSON only with this schema:
{{"title": "string", "summary": "string", "segments": [{{"speaker": "Host", "text": "string"}}]}}

SOURCE:
{source_text[:50000]}
"""

    try:
        response = gemini_generate_content(client, caller=__file__,
            model=model,
            contents=prompt,
            config={
                "temperature": temperature,
                "response_mime_type": "application/json",
            },
        )

        raw = response.text or ""
        if not raw:
            return None

        return json.loads(raw)
    except Exception as e:
        print(f"Generation error: {e}", file=sys.stderr)
        return None


def extract_facts_from_script(script: dict | list) -> list[Fact]:
    """Extract individual facts from a podcast script."""
    facts = []

    # Handle malformed responses (list instead of dict)
    if isinstance(script, list):
        # Try to extract from list items
        segments = script
    else:
        segments = script.get("segments", [])

    for segment in segments:
        # Handle different segment formats
        if isinstance(segment, str):
            text = segment.strip()
        elif isinstance(segment, dict):
            text = segment.get("text", "").strip()
        else:
            continue

        if not text:
            continue

        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", text)

        for sentence in sentences:
            sentence = sentence.strip()
            # Filter out short or generic sentences
            if len(sentence) < 30:
                continue
            if is_filler_sentence(sentence):
                continue

            facts.append(Fact(text=sentence))

    return facts


def is_filler_sentence(text: str) -> bool:
    """Check if a sentence is filler/transition rather than factual content."""
    filler_patterns = [
        r"^(?:welcome|hello|hi|hey)\b",
        r"^(?:let's|now|so|well|okay|alright)\b",
        r"^(?:today|in this|on this)\b",
        r"\?$",  # Questions
        r"^(?:thank|thanks)\b",
        r"^(?:that's (?:right|true|correct|interesting|fascinating))",
        r"^(?:absolutely|exactly|indeed|definitely)",
    ]

    lower = text.lower()
    for pattern in filler_patterns:
        if re.search(pattern, lower):
            return True

    return False


def write_outputs(ranked_facts: list[RankedFact], output_dir: Path) -> None:
    """Write ranking results to output files."""
    # JSON output with full details
    json_output = {
        "total_facts": len(ranked_facts),
        "tier_counts": {
            "tier_1_essential": sum(1 for rf in ranked_facts if rf.tier == 1),
            "tier_2_important": sum(1 for rf in ranked_facts if rf.tier == 2),
            "tier_3_supporting": sum(1 for rf in ranked_facts if rf.tier == 3),
            "tier_4_comprehensive": sum(1 for rf in ranked_facts if rf.tier == 4),
        },
        "facts": [
            {
                "rank": i + 1,
                "text": rf.fact.text,
                "hash": rf.fact.hash,
                "score": round(rf.score, 4),
                "tier": rf.tier,
                "frequency": round(rf.frequency, 4),
                "sources": rf.sources,
            }
            for i, rf in enumerate(ranked_facts)
        ],
    }

    (output_dir / "ranks.json").write_text(
        json.dumps(json_output, indent=2), encoding="utf-8"
    )

    # Markdown report
    md_lines = [
        "# Fact Ranking Report",
        "",
        f"**Total Facts:** {len(ranked_facts)}",
        "",
        "## Tier Distribution",
        "",
        f"- **Tier 1 (Essential):** {json_output['tier_counts']['tier_1_essential']}",
        f"- **Tier 2 (Important):** {json_output['tier_counts']['tier_2_important']}",
        f"- **Tier 3 (Supporting):** {json_output['tier_counts']['tier_3_supporting']}",
        f"- **Tier 4 (Comprehensive):** {json_output['tier_counts']['tier_4_comprehensive']}",
        "",
        "## Top 20 Facts",
        "",
        "| Rank | Score | Tier | Fact |",
        "|------|-------|------|------|",
    ]

    for i, rf in enumerate(ranked_facts[:20]):
        fact_preview = rf.fact.text[:100] + "..." if len(rf.fact.text) > 100 else rf.fact.text
        fact_preview = fact_preview.replace("|", "\\|")
        md_lines.append(f"| {i+1} | {rf.score:.3f} | {rf.tier} | {fact_preview} |")

    md_lines.extend([
        "",
        "## All Facts by Tier",
        "",
    ])

    for tier in range(1, 5):
        tier_facts = [rf for rf in ranked_facts if rf.tier == tier]
        tier_names = {1: "Essential", 2: "Important", 3: "Supporting", 4: "Comprehensive"}
        md_lines.append(f"### Tier {tier}: {tier_names[tier]} ({len(tier_facts)} facts)")
        md_lines.append("")

        for rf in tier_facts[:10]:  # Show top 10 per tier
            md_lines.append(f"- {rf.fact.text[:150]}{'...' if len(rf.fact.text) > 150 else ''}")

        if len(tier_facts) > 10:
            md_lines.append(f"- *...and {len(tier_facts) - 10} more*")
        md_lines.append("")

    (output_dir / "analysis.md").write_text("\n".join(md_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
