"""v1.61 prompts for SlideAnalysisEngine.

Two prompt modes:

* ``SLIDE_ANALYSIS_JSON_SYSTEM`` — primary path for instruction-tuned VLMs
  (Qwen3-VL, Pixtral, FastVLM-7B-MLX) that natively emit JSON.
* ``SLIDE_ANALYSIS_MARKDOWN_SYSTEM`` — fallback path that asks the VLM to
  produce a strict markdown layout the engine can re-parse without JSON.
"""

from __future__ import annotations


SLIDE_ANALYSIS_JSON_SYSTEM = (
    "You analyze a single conference slide image and return ONLY a JSON object "
    "matching this schema:\n"
    "{\n"
    '  "salient_claims": [string],\n'
    '  "figures": [{"caption": string, "modality": "chart"|"diagram"|"image"|"table"|"code"|"equation"|"other", "axes": [string] | null}],\n'
    '  "code_blocks": [string],\n'
    '  "equations": [string],\n'
    '  "headings": [string],\n'
    '  "body_text": string,\n'
    '  "reading_order_hint": string,\n'
    '  "confidence": number_between_0_and_1\n'
    "}\n"
    "Do not wrap the JSON in code fences or commentary. Preserve visible "
    "evidence and provenance (paper titles, citations, figure numbers, "
    "session names) verbatim where present. If a field has no content, "
    "use an empty array or empty string."
)

SLIDE_ANALYSIS_MARKDOWN_SYSTEM = (
    "You analyze a single conference slide image and return STRICT markdown "
    "with the following section headers (each present, in this order):\n"
    "## Salient Claims\n"
    "(bullet list, one claim per line, '- ' prefix)\n"
    "## Figures\n"
    "(bullet list of 'caption — modality [— axes if chart]', '- ' prefix)\n"
    "## Code Blocks\n"
    "(fenced ``` blocks, one per code region)\n"
    "## Equations\n"
    "(bullet list, one equation per line, '- ' prefix)\n"
    "## Headings\n"
    "(bullet list, '- ' prefix)\n"
    "## Body Text\n"
    "(plain paragraph)\n"
    "## Reading Order\n"
    "(plain text describing the visual flow)\n"
    "## Confidence\n"
    "(single decimal number between 0 and 1 on its own line)\n"
    "Preserve provenance (paper titles, citations, figure numbers, session "
    "names) verbatim. If a section has no content, leave the section body empty."
)


def build_user_prompt(*, task: str, prompt_overrides: dict | None = None) -> str:
    """Build the user-facing slide-analysis instruction for one call.

    ``task`` is the ``vision.*`` task id (e.g. ``vision.slide_analysis``).
    ``prompt_overrides`` allows callers to slot in extra emphasis (e.g.
    "describe the chart in detail" for ``vision.chart_summary``).
    """
    overrides = prompt_overrides or {}
    extra = overrides.get("emphasis", "").strip()
    base = (
        f"Analyze this slide as task {task}. "
        "Return your structured analysis in the format specified by the system prompt."
    )
    return f"{base} {extra}".strip() if extra else base
