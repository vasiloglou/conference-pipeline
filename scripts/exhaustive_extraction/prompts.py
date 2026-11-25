"""
Extraction prompt templates for NeurIPS 2025 content extraction.

These prompts guide the LLM to extract specific, factual content with
segment-bound timestamps and slide references.

Simplified prompt design (v2):
- Compact topic definitions (~1.5K chars vs 4K)
- Consolidated guardrails (3 rules vs 7)
- Minimal JSON example
"""

# Compact topic definitions - one line each
TOPIC_DEFINITIONS = """
01: Tools/Infrastructure - OSS tools, frameworks, benchmarks, deployment (vLLM, MLX, SWE-Bench, agent frameworks)
02: GenAI Architectures - Tokenization, attention mechanisms, scaling techniques, diffusion models, novel networks
03: Data Prep/Distillation - Data cleaning, compression, dataset distillation, smaller high-quality datasets
04: Data Attribution - Token attribution to sources, provenance tracking, source identification
05: Data Markets - Data valuation, compensation mechanisms, marketplace infrastructure, contribution tracking
06: Visual Models - Computer vision breakthroughs, new architectures, industry-impacting advances
07: Tabular/Relational - GNNs, relational transformers, TabPFN, text-to-SQL, predictive query systems
08: Language Models - LLM releases, reasoning advances, emergent properties, scaling insights, training techniques
09: GenAI in Science - Physics, chemistry, biology, drug discovery applications of generative AI
10: Serendipitous - Unexpected, creative, potentially disruptive ideas, odd applications, novel use cases
11: Ethics/Concerns - AI harm, discrimination, privacy, fairness issues, societal implications
12: Commercial Apps - Novel commercial applications, industry deployment patterns, business innovations
13: Industry Insights - Company developments, production systems, industry vs academic approaches
14: Synthetic Data - Generation methods, simulation techniques, data quality, GenAI for simulation
15: NeurIPS Happenings - Opening ceremony, community announcements, conference themes
16: Awards - Test of Time, Best Paper, Outstanding Paper, workshop awards
17: Scaling/Efficiency - Scaling laws, quantization, MoE, efficient attention, emergence, reasoning mechanisms
18: Agentic AI - Agent architectures, multi-agent systems, planning, tool use, code generation, robotics
19: Notable Quotes - Provocative, forward-looking statements, bold claims, paradigm shifts (min 12 words)
20: Small LMs - SLM advances (Phi, Gemma), distillation, speculative decoding, SLMs as building blocks
"""


def build_presentation_extraction_prompt(
    presentation_title: str,
    segment_index: int,
    segment_start_sec: float,
    segment_end_sec: float,
    segment_start_hms: str,
    segment_end_hms: str,
    segment_duration_sec: float,
    transcript_text: str,
    slides_list_with_ocr: str,
    topic_guardrails: str = ""
) -> str:
    """Build extraction prompt for a presentation segment."""
    return f"""Analyze this NeurIPS 2025 presentation segment and extract relevant content.

SEGMENT: "{presentation_title}" | Index {segment_index} | {segment_start_hms}-{segment_end_hms} ({segment_duration_sec:.0f}s)

TRANSCRIPT:
{transcript_text}

SLIDES:
{slides_list_with_ocr}

{f"TOPIC CUES: {topic_guardrails}" if topic_guardrails else ""}

RULES:
1. TIMESTAMPS: Use segment bounds for ALL extractions: start={segment_start_sec}, end={segment_end_sec}
2. QUOTE ANCHORS: Provide first 5-8 words and last 5-8 words of the relevant passage from TRANSCRIPT. We extract the full quote using these anchors.
3. OUTPUT: If no clear evidence for a topic, return empty array. Only cite slides from list above.

TOPICS:
{TOPIC_DEFINITIONS}

Return JSON:
{{"extractions": [{{"topic_id": "01", "topic_name": "...", "timestamp_start_sec": {segment_start_sec}, "timestamp_end_sec": {segment_end_sec}, "timestamp_start": "{segment_start_hms}", "timestamp_end": "{segment_end_hms}", "quote_start_anchor": "first five to eight words", "quote_end_anchor": "last five to eight words", "extraction_type": "verbatim_quote", "summary": "...", "slide_references": [{{"slide_id": "slide_042", "ocr_confidence": "high"}}], "relevance_score": 0.95, "transcript_quality": "high"}}]}}

Return ONLY valid JSON."""


def build_paper_extraction_prompt(
    paper_title: str,
    paper_abstract: str,
    paper_id: str
) -> str:
    """Build extraction prompt for a paper abstract."""
    return f"""Analyze this NeurIPS 2025 paper abstract and extract relevant content.

PAPER: "{paper_title}" | ID: {paper_id}

ABSTRACT:
{paper_abstract}

RULES:
1. QUOTE ANCHORS: Provide first 5-8 words and last 5-8 words of the relevant passage from ABSTRACT. We extract the full quote using these anchors.
2. OUTPUT: If no clear evidence for a topic, return empty array.

TOPICS:
{TOPIC_DEFINITIONS}

Return JSON:
{{"extractions": [{{"topic_id": "08", "topic_name": "...", "quote_start_anchor": "first five to eight words", "quote_end_anchor": "last five to eight words", "extraction_type": "verbatim_quote", "summary": "...", "relevance_score": 0.90, "source_coverage": "abstract_only"}}]}}

Return ONLY valid JSON."""


def build_paper_batch_extraction_prompt(
    papers: list  # List of {"paper_id": str, "paper_title": str, "paper_abstract": str}
) -> str:
    """Build extraction prompt for multiple paper abstracts in one call."""
    paper_sections = []
    for p in papers:
        paper_sections.append(f"""=== PAPER {p["paper_id"]} ===
Title: {p["paper_title"]}
Abstract: {p["paper_abstract"]}""")

    papers_text = "\n\n".join(paper_sections)

    return f"""Analyze these NeurIPS 2025 paper abstracts and extract relevant content from each.

{papers_text}

RULES:
1. QUOTE ANCHORS: For each extraction, provide first 5-8 words and last 5-8 words from that paper's ABSTRACT.
2. PAPER_ID: Include the paper_id in each extraction to identify which paper it came from.
3. OUTPUT: If no clear evidence for a topic in a paper, skip it. Return empty array if no papers have relevant content.

TOPICS:
{TOPIC_DEFINITIONS}

Return JSON with extractions from ALL papers combined:
{{"extractions": [{{"paper_id": "ABC123", "topic_id": "08", "topic_name": "...", "quote_start_anchor": "first five to eight words", "quote_end_anchor": "last five to eight words", "extraction_type": "verbatim_quote", "summary": "...", "relevance_score": 0.90, "source_coverage": "abstract_only"}}]}}

Return ONLY valid JSON."""


def build_workshop_extraction_prompt(
    workshop_title: str,
    talk_id: str,
    talk_title: str,
    talk_speaker: str,
    segment_start_hms: str,
    segment_end_hms: str,
    segment_duration_sec: float,
    transcript_text: str,
    slides_list_with_ocr: str,
    topic_guardrails: str = ""
) -> str:
    """Build extraction prompt for a workshop segment with talk context."""
    # Parse HH:MM:SS to seconds
    start_parts = segment_start_hms.split(":")
    segment_start_sec = int(start_parts[0]) * 3600 + int(start_parts[1]) * 60 + int(start_parts[2])
    end_parts = segment_end_hms.split(":")
    segment_end_sec = int(end_parts[0]) * 3600 + int(end_parts[1]) * 60 + int(end_parts[2])

    return f"""Analyze this NeurIPS 2025 workshop segment and extract relevant content.

WORKSHOP: "{workshop_title}"
TALK: "{talk_title}" ({talk_id}) by {talk_speaker}
SEGMENT: {segment_start_hms}-{segment_end_hms} ({segment_duration_sec:.0f}s)

TRANSCRIPT:
{transcript_text}

SLIDES:
{slides_list_with_ocr}

{f"TOPIC CUES: {topic_guardrails}" if topic_guardrails else ""}

RULES:
1. TIMESTAMPS: Use segment bounds for ALL extractions: start={segment_start_sec}, end={segment_end_sec}
2. QUOTE ANCHORS: Provide first 5-8 words and last 5-8 words of the relevant passage from TRANSCRIPT. We extract the full quote using these anchors.
3. OUTPUT: If no clear evidence for a topic, return empty array. Only cite slides from list above.

TOPICS:
{TOPIC_DEFINITIONS}

Return JSON:
{{"extractions": [{{"topic_id": "01", "topic_name": "...", "timestamp_start_sec": {segment_start_sec}, "timestamp_end_sec": {segment_end_sec}, "timestamp_start": "{segment_start_hms}", "timestamp_end": "{segment_end_hms}", "quote_start_anchor": "first five to eight words", "quote_end_anchor": "last five to eight words", "extraction_type": "verbatim_quote", "summary": "...", "slide_references": [{{"slide_id": "slide_042", "ocr_confidence": "high"}}], "relevance_score": 0.95, "transcript_quality": "high"}}]}}

Return ONLY valid JSON."""


def format_guardrails_for_prompt(guardrails: dict) -> str:
    """Format guardrails dict for inclusion in prompts."""
    if not guardrails:
        return ""
    lines = []
    for topic_id, rules in sorted(guardrails.items()):
        cues = ", ".join(rules.get("required_cues", [])) if rules.get("required_cues") else "none"
        min_q = rules.get("min_quote_len")
        min_s = rules.get("min_summary_len")
        allow_generic = rules.get("allow_generic", True)
        parts = [f"Topic {topic_id}: {cues}"]
        if min_q:
            parts.append(f"min_quote={min_q}")
        if min_s:
            parts.append(f"min_summary={min_s}")
        if not allow_generic:
            parts.append("no_generic")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def format_slides_list(slides_data: list) -> str:
    """Format slides list with OCR content for prompt (compact version)."""
    if not slides_data:
        return "No slides."

    formatted = []
    for slide in slides_data:
        slide_id = slide.get('slide_id', 'unknown')
        slide_text = slide.get('slide_text', '[No OCR]')
        # Truncate long OCR text
        if len(slide_text) > 500:
            slide_text = slide_text[:500] + "..."
        formatted.append(f"[{slide_id}] {slide_text}")

    return "\n".join(formatted)


def seconds_to_hms(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{mins:02d}:{secs:02d}"
