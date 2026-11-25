"""GPT Researcher hybrid analysis step with an injected researcher boundary."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from conference_pipeline.analysis._digest import sha256_file

REFERENCE_QUALITY_PROMPT = """
CRITICAL INSTRUCTION FOR CITATIONS:

When writing this report, you MUST include precise, traceable references for every claim.
For each finding, include:

1. **Source Details Block** with:
   - Presentation: [Exact presentation title]
   - Presentation ID: [ID from metadata, e.g., deep_learning_code_agentic]
   - Timestamp: [HH:MM:SS - HH:MM:SS] when discussed
   - Slide: [slide number if from slides]
   - File paths when available

2. **Verbatim Quote** - Include the exact quote from the source in a blockquote:
   > "The exact words from the transcript or slide"

3. **Source Type** - Mark each source as:
   - [LOCAL] - From the local conference vector store (has transcript/slide reference)
   - [WEB] - From Tavily search (needs local traceability check)

Example format for each finding:

### [Tool/Concept Name]

**Summary:** [Description]

**Source Details:**
- Presentation: "Deep Learning for Code in the Agentic Era"
- Presentation ID: deep_learning_code_agentic
- Timestamp: 43:39-47:17
- Type: [LOCAL]

**Verbatim Quote:**
> "OpenHands is an open-source AI software engineer that can..."

If information comes from web search only, note:
- Source: [WEB - needs local verification]
- URL: [the source URL]
- Action: Check if this content exists in local conference data

DO NOT make claims without citing the specific source with these details.
"""


class Researcher(Protocol):
    async def run(self, query: str, *, context_path: Path | None = None) -> str: ...


@dataclass(frozen=True)
class GPTResearcherInputs:
    workspace_root: Path
    vector_store_dir: Path
    output_dir: Path
    topics: tuple[str, ...]
    overwrite_existing: bool = True
    conference_label: str = "the conference"


@dataclass(frozen=True)
class GPTResearcherOutputs:
    report_paths: tuple[Path, ...]
    digest: str


def _topic_query(topic: str, *, conference_label: str = "the conference") -> str:
    title = topic[3:].replace("_", " ") if len(topic) > 3 and topic[:2].isdigit() else topic
    return f"{conference_label} analysis sweep: {title}"


async def _write_reports(inputs: GPTResearcherInputs, researcher: Researcher) -> tuple[Path, ...]:
    inputs.output_dir.mkdir(parents=True, exist_ok=True)
    report_paths: list[Path] = []
    for topic in inputs.topics:
        path = inputs.output_dir / f"{topic}.md"
        if path.exists() and not inputs.overwrite_existing:
            report_paths.append(path)
            continue
        body = await researcher.run(
            _topic_query(topic, conference_label=inputs.conference_label),
            context_path=inputs.vector_store_dir,
        )
        path.write_text(body.rstrip() + "\n", encoding="utf-8")
        report_paths.append(path)
    return tuple(report_paths)


def _combined_digest(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for item in sorted(sha256_file(path) for path in paths):
        digest.update(item.encode("ascii"))
    return digest.hexdigest()


def step_gpt_researcher_hybrid(
    inputs: GPTResearcherInputs,
    *,
    researcher: Researcher,
) -> GPTResearcherOutputs:
    try:
        report_paths = asyncio.run(_write_reports(inputs, researcher))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            report_paths = loop.run_until_complete(_write_reports(inputs, researcher))
        finally:
            loop.close()
    return GPTResearcherOutputs(report_paths=report_paths, digest=_combined_digest(report_paths))
