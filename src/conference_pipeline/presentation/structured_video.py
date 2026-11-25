"""Deterministic video builder for structured presentation delivery contracts."""

from __future__ import annotations

import html
import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from conference_pipeline.conference_identity import conference_name
from conference_pipeline.presentation.brand_template import (
    BRAND_ASSETS,
    BRAND_REFERENCE_NAME,
    CSS_VARIABLES,
    PROJECT_ROOT as BRAND_PROJECT_ROOT,
    RENDERER_GEOMETRY,
    brand_assets_sha256,
)
from conference_pipeline.presentation.display_text import (
    repair_leaked_macro_names,
    strip_event_kind_prefix,
    strip_markdown,
    unwrap_text_macros,
)
from conference_pipeline.presentation.legacy_markdown_gate import collect_legacy_markdown_paths
from conference_pipeline.presentation.source_inventory import is_locator_label
from conference_pipeline.presentation.source_links import SourceLinkResolver, canonical_public_url
from conference_pipeline.workflow_runs import write_json_atomic

DEFAULT_VIEWPORT = (1280, 720)
SVG_RAW_SOURCE_ID_RE = re.compile(
    r"\b(?:speech_transcript|paper_abstract|poster|openreview):[A-Za-z0-9_.:/-]+"
    # Shortened provenance forms the model sometimes invents for footers
    # (e.g. "transcript:016fd...seg00000", "paper:0CajQNVKyB"). An id carries a
    # digit; "Position paper: static cues" is prose (ICML 2026 deck 06).
    r"|\b(?:transcript|paper|segment|seg)\s*:\s*(?=[A-Za-z0-9_.…:/-]*\d)[A-Za-z0-9][A-Za-z0-9_.…:/-]{3,}"
)
SVG_INTERNAL_SCAFFOLD_TERMS = (
    "source contribution -> visual proof object -> viewer-facing claim",
    "source contribution",
    "visual proof object",
    "viewer-facing claim",
    "source evidence -> visual mechanism -> audience claim",
    "source evidence ->",
    "visual mechanism ->",
    "source evidence",
    "visual mechanism",
    "audience claim",
    "evidence maps to deployment action",
    "named study -> slide claim -> mechanism view",
    "named study -> control layer -> deployment decision",
    "deployment takeaway",
    "source record",
    "phase 16",
    "phase16",
    "render_spec",
    "cluster_id",
)
# The pieces of those two chains, on their own. Each is ordinary subject
# matter -- a deck about camera control says "control layer", a deck about
# deployment says "deployment decision" -- so one is content and several
# together are a leaked scaffold. Listing them individually refused a figure
# for saying what its slide was about (deck 05, sl_18).
SVG_SCAFFOLD_FRAGMENTS = (
    "named study",
    "slide claim",
    "mechanism view",
    "control layer",
    "deployment decision",
)
SVG_SCAFFOLD_FRAGMENT_QUORUM = 2
# Per-column item budget for the two-column bookend layout, matching the
# TOP_EVENTS / TOP_PAPERS caps that add_bookend_slides.py applies when it
# builds one. Topic decks stay under it; an executive topic index does not,
# which is what the itinerary layout exists for.
BOOKEND_COLUMN_MAX_ITEMS = 8

SVG_FRAGMENT_END_WORDS = frozenset(
    {
        # "4th Structured Probabilistic Inference &" shipped on ICML 2026 deck 01:
        # a connector mark ends a cut label exactly as a stopword does.
        "&",
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "because",
        "for",
        "from",
        "in",
        "nor",
        "of",
        "or",
        "the",
        "to",
        "with",
        "within",
        # A citation cut mid-phrase reads as broken, and these end a line only
        # when something was dropped after them. Found on deck 08's slide 22,
        # whose last source line ended "... via".
        "via",
        "using",
        "through",
        "by",
        "per",
        "into",
        "onto",
        "upon",
        "toward",
        "towards",
        "between",
    }
)


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _segment_duration(segment: dict[str, Any]) -> float:
    start = float(segment.get("start_seconds") or 0)
    end = float(segment.get("end_seconds") or 0)
    duration = end - start
    if duration <= 0:
        duration = float(segment.get("duration_seconds") or 1.0)
    return max(duration, 0.25)


def _run(cmd: list[str]) -> None:
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{cmd[0]} failed with exit {completed.returncode}: {completed.stderr.strip()}"
        )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _isoformat_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _html_relative_src(path: Path, deck_dir: Path) -> str:
    return os.path.relpath(path.resolve(), deck_dir.resolve()).replace(os.sep, "/")


def _infer_workspace(path: Path) -> Path:
    parts = path.resolve().parts
    if "decks" in parts:
        index = parts.index("decks")
        return Path(*parts[:index])
    return path.parent


def _slide_title(slide: dict[str, Any], index: int) -> str:
    blocks = slide.get("layout_blocks")
    if isinstance(blocks, list):
        for block in blocks:
            if isinstance(block, dict) and str(block.get("text") or "").strip():
                return str(block["text"]).strip()
    return str(slide.get("title") or slide.get("slide_id") or f"Slide {index}").strip()


def _asset_src(raw_ref: str, *, deck_dir: Path, workspace: Path) -> str:
    return _html_relative_src(_resolve_asset_path(raw_ref, deck_dir=deck_dir, workspace=workspace), deck_dir)


def _resolve_asset_path(raw_ref: str, *, deck_dir: Path, workspace: Path) -> Path:
    raw_path = Path(raw_ref)
    if raw_path.is_absolute():
        return raw_path.resolve()
    elif raw_ref.startswith("decks/"):
        return (workspace / raw_path).resolve()
    return (deck_dir / raw_path).resolve()


def _source_href(raw_ref: str, *, deck_dir: Path, workspace: Path) -> str:
    raw_ref = raw_ref.strip()
    if not raw_ref:
        return ""
    if raw_ref.startswith(("http://", "https://", "mailto:", "#")):
        return raw_ref
    raw_path = Path(raw_ref)
    if raw_path.is_absolute():
        return _html_relative_src(raw_path, deck_dir)
    workspace_candidate = (workspace / raw_path).resolve()
    if workspace_candidate.exists():
        return _html_relative_src(workspace_candidate, deck_dir)
    project_candidate = (BRAND_PROJECT_ROOT / raw_path).resolve()
    if project_candidate.exists():
        return _html_relative_src(project_candidate, deck_dir)
    return raw_ref


def _safe_review_href(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    public = canonical_public_url(raw)
    if public:
        return public
    lowered = raw.lower()
    if lowered.startswith(("file:", "data:", "javascript:", "mailto:")):
        return ""
    if re.match(r"^[a-z][a-z0-9+.-]*:", lowered):
        return ""
    if raw.startswith(("/", "\\")):
        return ""
    if raw.startswith(("./", "../")) or "/" in raw:
        return raw
    return ""


def _render_source_chips(
    source_chips: list[object],
    *,
    deck_dir: Path,
    workspace: Path,
    link_resolver: SourceLinkResolver | None = None,
) -> str:
    chips: list[str] = []
    seen_labels: set[str] = set()
    resolver = link_resolver or SourceLinkResolver.from_workspace(workspace)
    for reference in source_chips[:4]:
        if not isinstance(reference, dict):
            continue
        label = _source_chip_label(reference)
        # Two ledger entries for the same work (e.g. two transcript segments of
        # one talk) must not render as duplicate pills.
        if label.lower() in seen_labels:
            continue
        seen_labels.add(label.lower())
        source_id = str(reference.get("source_id") or reference.get("label") or "").strip()
        # Source chips should link to public source pages only. Local ledger,
        # cache, and media paths remain provenance, not viewer-facing anchors.
        links = resolver.resolve_links(reference)
        # A poster always corresponds to a paper: the chip names the work and
        # links the paper; the poster page rides along as a small second link.
        url = links.get("paper") or links.get("primary") or links.get("poster") or ""
        poster_url = links.get("poster", "")
        if poster_url == url:
            poster_url = ""
        if url:
            chips.append(
                '<a class="source-chip" data-source-id="{source_id}" href="{url}" target="_blank" rel="noopener">{label}</a>'.format(
                    source_id=html.escape(source_id, quote=True),
                    url=html.escape(url, quote=True),
                    label=html.escape(label),
                )
            )
            if poster_url:
                # The poster page rides inside the same pill as a second link
                # ("Title · poster"); a separate tiny pill read as an unfinished
                # placeholder in pixel review.
                # A recorded talk or oral that also has a program page is not
                # thereby a poster (ICML 2026 deck 12: an Oral 6F position talk).
                is_poster = str(reference.get("source_type") or "").lower() in {"poster", "paper", "paper_abstract", "paper_text"}
                chips[-1] = (
                    '<span class="source-chip source-chip-dual" data-source-id="{source_id}">'
                    '<a href="{url}" target="_blank" rel="noopener">{label}</a>'
                    '<a class="source-chip-poster" href="{poster}" target="_blank" rel="noopener" title="{title}">· {text}</a>'
                    "</span>"
                ).format(
                    source_id=html.escape(source_id, quote=True),
                    url=html.escape(url, quote=True),
                    label=html.escape(label),
                    poster=html.escape(poster_url, quote=True),
                    title="Poster page" if is_poster else "Program page",
                    text="poster" if is_poster else "program page",
                )
        else:
            chips.append(
                '<span class="source-chip" data-source-id="{source_id}">{label}</span>'.format(
                    source_id=html.escape(source_id, quote=True),
                    label=html.escape(label),
                )
            )
    if not chips:
        chips.append('<span class="source-chip">Source-grounded evidence</span>')
    return '<div class="reference-chips" data-qc-role="source-chips"><span>Sources</span>{}</div>'.format(
        "".join(chips)
    )


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


_LOCATOR_KIND_RE = re.compile(r"^(poster|oral|spotlight|session|talk|paper|workshop|tutorial|event|item)[-_ ]?(\d+)$", re.I)


def _humanize_locator(value: str) -> str:
    match = _LOCATOR_KIND_RE.match(" ".join(str(value or "").split()))
    if match:
        return f"{match.group(1).capitalize()} {match.group(2)}"
    return "Conference talk"


_INLINE_MATH_RE = re.compile(r"\$([^$]{1,40})\$")
_MATH_BRACES_RE = re.compile(r"[{}]")


def strip_inline_math(value: object) -> str:
    """Render a work title's inline LaTeX math as the plain text it stands for.

    Corpus titles arrive as their authors typed them, so a name like
    ``Part$^{2}$GS`` carries TeX markup that is meaningless to a slide: the
    pixel judge reads it as a rendering defect, because that is exactly what it
    looks like. Superscript/subscript markers are dropped rather than
    transliterated -- ``Part2GS`` is the name people search for.
    """

    # ``$\\texttt{FlashSchNet}$`` is a name, not math: unwrap the text macro (and
    # repair a macro name an earlier cleaner already left glued to it) first, or
    # the braces go and ``textttFlashSchNet`` is what the audience reads.
    text = repair_leaked_macro_names(unwrap_text_macros(str(value or "")))
    if "$" not in text:
        return text
    text = _INLINE_MATH_RE.sub(lambda match: _MATH_BRACES_RE.sub("", match.group(1)), text)
    return text.replace("^", "").replace("_", "")


_PROPOSED_SYSTEM_RE = re.compile(
    r"^\s*(?:The|This)\s+(?:paper|work|talk|poster)\s+(?:introduces|proposes|presents)\s+"
    r"([A-Za-z0-9][A-Za-z0-9²³+\-]*)(?=[\s,.;:(])"
)


def _proposed_system_name(note: object) -> str:
    """The system a contribution note says the work introduces, when it is a name.

    Only a single product-shaped token qualifies (two or more capitals, e.g.
    ``SandboxVLM``, ``FLARE``, ``S²-MLLM``); "introduces a framework ..." never does.
    """

    match = _PROPOSED_SYSTEM_RE.match(str(note or ""))
    if not match:
        return ""
    name = match.group(1).rstrip("-")
    if not 3 <= len(name) <= 24 or sum(ch.isupper() for ch in name) < 2:
        return ""
    return name


def _superscript_folded(text: str) -> str:
    # Notes spell "S2-MLLM" where titles print "S²-MLLM"; both name one system.
    return text.lower().replace("²", "2").replace("³", "3")


def _source_chip_label(reference: dict[str, Any]) -> str:
    people = [
        item
        for key in ("speakers", "authors")
        for item in _string_list(reference.get(key))
    ]
    named_entities = [
        strip_inline_math(item)
        for item in _string_list(reference.get("systems")) + _string_list(reference.get("datasets"))
    ]
    works = [
        strip_event_kind_prefix(strip_inline_math(work))
        for work in _string_list(reference.get("works"))
        if not is_locator_label(work)
    ]
    title = strip_event_kind_prefix(
        strip_inline_math(reference.get("title") or reference.get("label") or "Source")
    ).strip()
    source_label = str(reference.get("label") or "").strip()
    if is_locator_label(title):
        # "poster-63306" names WHERE the work was presented, not the work; fall
        # back to any real title we have, else present the locator as a phrase.
        title = works[0] if works else _humanize_locator(title)
    if not named_entities:
        # A paper whose title never names its system ("Abstract 3D Perception
        # for Spatial Intelligence..." introduces SandboxVLM) left the chip
        # without the name the slide and narration use, and Gate A reviewers
        # read the slide as uncited (CVPR 2026 deck 04, 2026-11-10).
        proposed = _proposed_system_name(reference.get("contribution_note"))
        if proposed and _superscript_folded(proposed) not in _superscript_folded(title):
            named_entities = [proposed]
    if named_entities:
        base = named_entities[0]
    elif title and title != source_label and title.lower() != "source":
        base = title
    elif works:
        base = works[0]
    else:
        base = title
    base = _complete_short_title(base)
    # A work's natural short name is its pre-colon head ("Hybrid Reinforcement:
    # when reward is..." -> "Hybrid Reinforcement"). Mid-title word cuts produce
    # dangling fragments ("...when reward is") that read as rendering defects.
    head = base.split(":", 1)[0].strip()
    # ...and a one-token head is the short name when it is an acronym-shaped
    # system name ("UNI-OOD: Unified Object- and Image-level ..."), which a cut
    # at the length limit turned into "UNI-OOD: Unified Object-" on CVPR 2026.
    acronym_head = len(head.split()) == 1 and sum(ch.isupper() for ch in head) >= 2
    if head != base and ((len(head.split()) >= 2 and len(head) >= 8) or acronym_head):
        base = head
    if people:
        label = f"{people[0]}: {base}"
        # The work's name is the reference the audience can look up; a person
        # prefix that forces the title to be cut mid-phrase is worse than no
        # prefix ("Yujin Potter: Peer-Preservation in Frontier" -> the full title).
        if len(label) > 46 and len(base) <= 46:
            label = base
    else:
        label = base
    # A chip wraps to two lines, so a label a few characters over the limit
    # fits whole; cutting it dropped the noun ("AI for science, physics, and
    # autonomous" on the ICML 2026 executive deck).
    if len(label) <= 56:
        return label
    return _compact_label(label, limit=46)


_SOURCE_TITLE_SHORTCUTS: tuple[tuple[str, str], ...] = (
    ("clip behaves like a bag-of-words", "CLIP Bag-of-Words"),
    ("david bau talk and representational alignment", "Representational Alignment"),
    ("david bau talk: traces", "Traces of Meaning"),
    ("detecting misbehaviors of large", "Misbehavior Detection"),
    ("persona features control emergent", "Persona Features"),
    ("the price of amortized inference", "Amortized inference"),
    ("toward faithful rag", "Faithful RAG"),
    ("understanding language prior", "Language Prior"),
)


def _complete_short_title(value: object) -> str:
    text = " ".join(html.unescape(str(value or "")).split()).strip()
    lowered = text.lower()
    for needle, replacement in _SOURCE_TITLE_SHORTCUTS:
        if needle in lowered:
            return replacement
    text = re.sub(r"^[A-Z][A-Za-z .'-]+ talk and\s+", "", text)
    text = re.sub(r"^[A-Z][A-Za-z .'-]+ talk:\s*", "", text)
    return text


def _compact_label(value: object, *, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    shortened = text[:limit].rsplit(" ", 1)[0].strip()
    # Reuse the canonical fragment-end-word list (SVG_FRAGMENT_END_WORDS) so a
    # source-chip label gets the same dangling-preposition trim already applied
    # to figure labels and captions, rather than the narrower local set this
    # used to carry — that set was missing "via" and produced exactly the
    # dangling-fragment defect ("UVU: Improving Multimodal Understanding via")
    # the wider list was built to catch on deck 08's slide 22.
    words = shortened.split()
    while len(words) > 1 and _is_fragment_end_word(words[-1]):
        words.pop()
    # Never end on a hyphenated half-compound ("Unified Object-"). No ellipsis:
    # the visual judge fails every "..."-cut chip as clipped text, while a cut
    # at a word boundary reads as a short name.
    return (" ".join(words) or shortened).rstrip(".,;:-\u2013")


def _is_fragment_end_word(word: str) -> bool:
    """True when ``word`` is a genuine sentence-ending stopword/article.

    Excludes a single uppercase letter (``A``, ``B``, ``C``, ...): those are
    almost always a label (``Variant A``, ``Group B``, ``persona A``), not the
    English indefinite article ``a``. Comparing case-insensitively without this
    guard flags every label-ending line as an unfinished sentence fragment.
    """
    stripped = word.strip(".,;:!?")
    if len(stripped) == 1 and stripped.isalpha() and stripped.isupper():
        return False
    return stripped.lower() in SVG_FRAGMENT_END_WORDS


_COMPLETE_SENTENCE_HARD_CEILING = 300


def _compact_text(value: object, *, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    # A caption that is already one complete, correctly-punctuated sentence
    # (optionally quote-closed) reads far better slightly over the nominal
    # limit than mid-word truncated into a fragment ("...prompt inject...").
    # Only the hard ceiling below forces truncation of a genuine single
    # sentence; content that isn't terminal-punctuated at all (a run-on or
    # multi-sentence blob) still gets the normal truncate-to-limit treatment.
    if len(text) <= _COMPLETE_SENTENCE_HARD_CEILING and re.search(r"[.!?][\"'’”]?$", text):
        return text
    truncated = text[:limit].rsplit(" ", 1)[0].strip()
    # Never let a truncated caption trail off on a conjunction/preposition
    # ("...visual prompt injection, and."); reviewers read it as an unfinished
    # fragment (2026-06-19 Claude/Codex/Gemini panel review).
    words = truncated.rstrip(".,;:!?").split()
    while len(words) > 1 and _is_fragment_end_word(words[-1]):
        words.pop()
    cleaned = " ".join(words).rstrip(".,;:!?").strip()
    return f"{cleaned}." if cleaned else truncated + "."


def _visible_evidence_caption(value: object, *, executive_lesson: bool) -> str:
    text = " ".join(str(value or "").split())
    if executive_lesson:
        text = re.sub(
            r"^(Direct transcript quote|Tight transcript paraphrase|Transcript-backed evidence)\s*:\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
    return text


def _first_sentence(value: object) -> str:
    text = " ".join(str(value or "").split())
    # See the matching comment in generate_executive_overview_decks.first_sentence:
    # allow an optional closing quote between terminal punctuation and the
    # required whitespace/end, so a sentence ending on a quoted phrase isn't
    # skipped past in favor of the next one.
    match = re.search(r"(.+?[.!?][\"'’”]?)(?:\s|$)", text)
    return match.group(1) if match else text


def _shorten_caption_to_clause(text: str, *, max_chars: int) -> str:
    """Shorten an over-long caption to the largest prefix of complete
    sentences (then complete clauses) that fits ``max_chars``, ending cleanly.

    The left evidence rail is a narrow (~200px) column. A genuine outlier
    compound sentence (~290 chars) preserved whole by ``_compact_text`` wraps
    to ~12 tiny lines there and reads as illegible dense copy at presentation
    distance, even though the same text is fine in the wide figure card. This
    trims the *rail* copy to a readable excerpt without ever leaving a
    dangling ellipsis or a comma stranded before a closing quote.
    """
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= max_chars:
        return normalized

    def _clean_end(candidate: str) -> str:
        candidate = candidate.strip()
        # Drop a comma/semicolon left immediately before a closing quote
        # (", -> ") and any trailing clause punctuation.
        candidate = re.sub(r'[,;:—–]\s*(["”’])$', r"\1", candidate)
        return candidate.rstrip(",;:—– ").strip()

    sentences = re.split(r'(?<=[.!?"”’])\s+', normalized)
    for count in range(len(sentences), 0, -1):
        candidate = _clean_end(" ".join(sentences[:count]))
        if candidate and len(candidate) <= max_chars:
            return candidate
    first = sentences[0]
    clauses = re.split(r"(?<=[,;:—–])\s+", first)
    for count in range(len(clauses), 0, -1):
        candidate = _clean_end(" ".join(clauses[:count]))
        if candidate and len(candidate) <= max_chars:
            return candidate
    words = first.split()
    while len(words) > 1 and len(" ".join(words)) > max_chars:
        words.pop()
    return _clean_end(" ".join(words))


def _slide_copy_html(slide: dict[str, Any], heading: str = "") -> str:
    notes = str(slide.get("speaker_notes") or "")
    is_executive_lesson = bool(str(slide.get("executive_lesson_id") or "").strip())
    # Only texts over _COMPLETE_SENTENCE_HARD_CEILING (a genuine minority of
    # outlier compound sentences) ever hit this limit now that _compact_text
    # preserves a complete sentence whole up to the ceiling; raised from
    # 88/96 so that fallback still reads as a real clause, not a fragment.
    caption_limit = 154 if is_executive_lesson else 180
    takeaway_limit = 132 if is_executive_lesson else 140
    # The narrow rail can't carry a ~290-char outlier compound sentence
    # legibly (it wraps to ~12 tiny lines); trim it to a complete-clause
    # excerpt first. The figure card (wide) still shows the fuller text.
    rail_caption_max_chars = 210 if is_executive_lesson else 175
    caption = _compact_text(
        _shorten_caption_to_clause(
            _visible_evidence_caption(
                slide.get("evidence_caption") or _first_sentence(notes) or "Source-grounded evidence.",
                executive_lesson=is_executive_lesson,
            ),
            max_chars=rail_caption_max_chars,
        ),
        limit=caption_limit,
    )
    takeaway = _compact_text(
        slide.get("visual_takeaway") or "The figure anchors the claim in cited evidence.",
        limit=takeaway_limit,
    )
    source_anchor = _compact_text(slide.get("source_anchor") or "", limit=74)

    def _norm(text: str) -> str:
        return " ".join(text.lower().split()).rstrip(".")

    # The rail must not restate the slide heading (reviewers read the echo as a
    # duplication bug); drop duplicate lines and keep the rail for Sources.
    if heading and _norm(caption) == _norm(heading):
        caption = ""
    if not takeaway or _norm(takeaway) in {_norm(heading), _norm(caption)}:
        takeaway = ""
    parts = ['<div class="evidence-copy" data-qc-role="evidence-copy">']
    if caption:
        parts.append(f'<p class="evidence-caption">{html.escape(caption)}</p>')
    # For executive lessons the leadership decision is rendered inside the figure
    # ("Leadership decision" panel), so repeating it in the rail reads as a
    # duplication bug (2026-06-19 Claude/Codex/Gemini panel review). The rail then
    # carries the transcript insight + named sources only. Non-executive slides
    # keep their takeaway line.
    if takeaway and not is_executive_lesson:
        parts.append(f'<p class="visual-takeaway">{html.escape(takeaway)}</p>')
    if is_executive_lesson and source_anchor:
        parts.append(f'<p class="evidence-source">Source: {html.escape(source_anchor)}</p>')
    parts.append("</div>")
    return "".join(parts)


def _visual_legibility_findings(
    slides: list[dict[str, Any]],
    *,
    deck_dir: Path,
    workspace: Path,
) -> list[str]:
    findings: list[str] = []
    for slide in slides:
        slide_id = str(slide.get("slide_id") or "slide")
        for raw_ref in slide.get("visual_asset_refs", []):
            if not isinstance(raw_ref, str) or not raw_ref.strip():
                continue
            path = _resolve_asset_path(raw_ref, deck_dir=deck_dir, workspace=workspace)
            if path.suffix.lower() != ".svg" or not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            visible_text = _svg_visible_text(text)
            font_sizes = [
                float(value)
                for value in re.findall(r'font-size=["\']?([\d.]+)', text)
                + re.findall(r"font-size:\s*([\d.]+)(?:px|pt)?", text)
            ]
            # The renderer downscales the 1280x720 figure SVG to a ~930px panel
            # (~0.73x), so SVG-space sizes below 18px render illegibly small.
            tiny = [size for size in font_sizes if size < 18]
            text_nodes = re.findall(r"<text\b", text)
            if tiny:
                findings.append(f"{slide_id}:{path.name}:font_too_small:{min(tiny):.1f}")
            if len(text_nodes) > 36:
                findings.append(f"{slide_id}:{path.name}:too_many_text_labels:{len(text_nodes)}")
            lowered = visible_text.lower()
            fragments = sum(1 for term in SVG_SCAFFOLD_FRAGMENTS if term in lowered)
            if (any(term in lowered for term in SVG_INTERNAL_SCAFFOLD_TERMS)
                    or fragments >= SVG_SCAFFOLD_FRAGMENT_QUORUM):
                findings.append(f"{slide_id}:{path.name}:internal_scaffold_visible")
            if SVG_RAW_SOURCE_ID_RE.search(visible_text):
                findings.append(f"{slide_id}:{path.name}:raw_source_metadata_visible")
            stacked = _svg_stacked_text_positions(text)
            if stacked:
                findings.append(
                    f"{slide_id}:{path.name}:stacked_text_same_position:{stacked[0]}"
                )
            for mismatch in _svg_source_person_mismatches_positional(slide, text):
                findings.append(f"{slide_id}:{path.name}:{mismatch}")
            for mismatch in _svg_source_work_mismatches(slide, text):
                findings.append(f"{slide_id}:{path.name}:{mismatch}")
            for fragment in _svg_fragment_lines(visible_text):
                findings.append(f"{slide_id}:{path.name}:text_fragment:{fragment[:48]}")
    return findings


def _svg_stacked_text_positions(svg_text: str) -> list[str]:
    """Detect distinct text elements drawn at the same (x, y) — they render as an
    unreadable overlapping garble (e.g. a footer drawn twice on one anchor)."""

    positions: dict[tuple[int, int], set[str]] = {}
    for match in re.finditer(r"<text\b([^>]*)>(.*?)</text>", svg_text, flags=re.DOTALL | re.I):
        attrs = match.group(1)
        body = re.sub(r"<[^>]+>", " ", match.group(2))
        body = " ".join(html.unescape(body).split())
        if not body:
            continue
        x_match = re.search(r"\bx=[\"']?(-?[\d.]+)", attrs)
        y_match = re.search(r"\by=[\"']?(-?[\d.]+)", attrs)
        if not x_match or not y_match:
            continue
        key = (round(float(x_match.group(1))), round(float(y_match.group(1))))
        positions.setdefault(key, set()).add(body)
    return [f"{key[0]},{key[1]}" for key, bodies in sorted(positions.items()) if len(bodies) > 1]


def _svg_positioned_texts(svg_text: str) -> list[tuple[float, float, str]]:
    """All SVG text elements with explicit coordinates, as (x, y, content)."""

    out: list[tuple[float, float, str]] = []
    for match in re.finditer(r"<text\b([^>]*)>(.*?)</text>", svg_text, flags=re.DOTALL | re.I):
        attrs = match.group(1)
        body = re.sub(r"<[^>]+>", " ", match.group(2))
        body = " ".join(html.unescape(body).split())
        if not body:
            continue
        x_match = re.search(r"\bx=[\"']?(-?[\d.]+)", attrs)
        y_match = re.search(r"\by=[\"']?(-?[\d.]+)", attrs)
        if not x_match or not y_match:
            continue
        out.append((float(x_match.group(1)), float(y_match.group(1)), body))
    return out


def _svg_source_work_mismatches(slide: dict[str, Any], svg_text: str) -> list[str]:
    """Detect a named work tagged with a different work's SRC chip (attribution
    drift inside the figure). Pairing is by RENDERED COORDINATES — each bare
    SRC-NNN chip is attributed to the spatially nearest work title — because
    document-order adjacency crosses box boundaries and false-positives."""

    label_by_alias: dict[str, set[str]] = {}
    for chip in slide.get("source_chips", []):
        if not isinstance(chip, dict):
            continue
        label = str(chip.get("label") or chip.get("source_label") or "").strip().lower()
        if not label:
            continue
        candidates = [str(chip.get("title") or "")]
        candidates.extend(str(work) for work in chip.get("works", []) or [])
        for candidate in candidates:
            candidate = candidate.strip()
            head = candidate.split(":", 1)[0].strip()
            head_words = head.split()
            # Figures label nodes with SHORT work names ("Strategic Scaling"),
            # so 2- and 3-word prefixes must also resolve to the work's chips.
            for alias in (
                candidate,
                head,
                " ".join(head_words[:3]),
                " ".join(head_words[:2]),
            ):
                if len(alias) >= 8 and len(alias.split()) >= 2:
                    label_by_alias.setdefault(alias.lower(), set()).add(label)
    if not label_by_alias:
        return []
    texts = _svg_positioned_texts(svg_text)
    chips = [
        (x, y, body.strip().lower())
        for x, y, body in texts
        if re.fullmatch(r"SRC-\d{3}", body.strip(), flags=re.I)
    ]
    # A row of >=3 chips on one baseline is the footer ledger strip, not a tag
    # on a work box — exclude it from nearest-work attribution.
    baseline_counts: dict[int, int] = {}
    for _x, y, _label in chips:
        baseline_counts[round(y)] = baseline_counts.get(round(y), 0) + 1
    chips = [
        (x, y, label) for x, y, label in chips if baseline_counts[round(y)] < 3
    ]
    works: list[tuple[float, float, str]] = []
    for x, y, body in texts:
        if "->" in body:
            continue  # relationship edge labels name works without owning a chip
        lowered = body.lower()
        hits = [alias for alias in label_by_alias if alias in lowered]
        if hits:
            works.append((x, y, max(hits, key=len)))
    mismatches: set[str] = set()
    for chip_x, chip_y, chip_label in chips:
        nearest_alias: str | None = None
        nearest_distance: float | None = None
        for work_x, work_y, alias in works:
            distance = ((chip_x - work_x) ** 2 + (chip_y - work_y) ** 2) ** 0.5
            if distance <= 200 and (nearest_distance is None or distance < nearest_distance):
                nearest_distance = distance
                nearest_alias = alias
        if nearest_alias is None:
            continue  # footer / unattached chips have no nearby work title
        if chip_label not in label_by_alias[nearest_alias]:
            mismatches.add(f"source_work_mismatch:{nearest_alias[:32]}:{chip_label}")
    return sorted(mismatches)


def _svg_text_lines(svg_text: str) -> list[str]:
    """Extract visible SVG text nodes as separate normalized lines."""

    chunks: list[str] = []
    for match in re.finditer(r"<text\b[^>]*>(.*?)</text>", svg_text, flags=re.DOTALL | re.I):
        body = re.sub(r"<[^>]+>", " ", match.group(1))
        body = html.unescape(body)
        body = " ".join(body.split())
        if body:
            chunks.append(body)
    return chunks


def _svg_visible_text(svg_text: str) -> str:
    """Extract visible SVG text for deterministic pre-render checks."""

    return "\n".join(_svg_text_lines(svg_text))


def _svg_source_person_mismatches_positional(slide: dict[str, Any], svg_text: str) -> list[str]:
    """Coordinate-based person↔chip attribution: a person's name is judged by
    the spatially nearest SRC chip (footer ledger rows excluded), mirroring the
    work-mismatch check. The line-order variant false-positives on dense
    multi-box layouts."""

    label_people: dict[str, set[str]] = {}
    all_people: set[str] = set()
    for chip in slide.get("source_chips", []):
        if not isinstance(chip, dict):
            continue
        label = str(chip.get("label") or chip.get("source_label") or "").strip().lower()
        people = {
            str(person).strip()
            for field in ("authors", "speakers")
            for person in chip.get(field, [])
            if len(str(person).strip().split()) >= 2
        }
        if label and people:
            label_people[label] = people
            all_people |= people
    if not label_people:
        return []
    texts = _svg_positioned_texts(svg_text)
    chips = [
        (x, y, body.strip())
        for x, y, body in texts
        if re.fullmatch(r"SRC-\d{3}", body.strip(), flags=re.I)
    ]
    baseline_counts: dict[int, int] = {}
    for _x, y, _label in chips:
        baseline_counts[round(y)] = baseline_counts.get(round(y), 0) + 1
    chips = [(x, y, label) for x, y, label in chips if baseline_counts[round(y)] < 3]
    if not chips:
        return []
    mismatches: set[str] = set()
    for x, y, body in texts:
        named = [p for p in all_people if p.lower() in body.lower()]
        if not named:
            continue
        nearest_label: str | None = None
        nearest_distance: float | None = None
        for cx, cy, label in chips:
            distance = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            if distance <= 200 and (nearest_distance is None or distance < nearest_distance):
                nearest_distance = distance
                nearest_label = label
        if nearest_label is None:
            continue
        for person in named:
            if person not in label_people.get(nearest_label.lower(), set()):
                mismatches.add(f"source_person_mismatch:{nearest_label}:{person}")
    return sorted(mismatches)


def _svg_source_person_mismatches(slide: dict[str, Any], lines: list[str]) -> list[str]:
    source_records = _svg_source_records(slide)
    if not source_records or not lines:
        return []
    lowered_lines = [line.lower() for line in lines]
    people = sorted(
        {
            person
            for record in source_records
            for person in record["people"]
            if len(person.split()) >= 2
        },
        key=len,
        reverse=True,
    )
    mismatches: list[str] = []
    for index, line in enumerate(lowered_lines):
        if _is_svg_aggregate_people_line(lowered_lines, index):
            continue
        visible_people = [person for person in people if person.lower() in line]
        if not visible_people:
            continue
        for person in visible_people:
            nearest = _nearest_svg_source_records(source_records, lowered_lines, index)
            if not nearest:
                continue
            if any(person in record["people"] for record in nearest):
                continue
            labels = ",".join(record["label"] for record in nearest)
            mismatches.append(f"source_person_mismatch:{labels}:{person}")
    return mismatches


def _is_svg_aggregate_people_line(lowered_lines: list[str], index: int) -> bool:
    window = " ".join(lowered_lines[max(0, index - 1) : index + 1])
    aggregate_markers = ("contributors", "evidence authors", "sources:")
    return any(marker in window for marker in aggregate_markers)


def _svg_source_records(slide: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for chip in slide.get("source_chips", []):
        if not isinstance(chip, dict):
            continue
        label = str(chip.get("label") or chip.get("source_label") or "").strip()
        people = {
            str(person).strip()
            for field in ("authors", "speakers")
            for person in chip.get(field, [])
            if str(person).strip()
        }
        if not label or not people:
            continue
        aliases = {label}
        for field in ("title", "contribution_note"):
            value = str(chip.get(field) or "").strip()
            if value:
                aliases.add(value)
                aliases.add(value.split(":", 1)[0].strip())
                aliases.add(value.split("—", 1)[0].strip())
        for field in ("systems", "works"):
            for value in chip.get(field, []):
                text = str(value).strip()
                if text:
                    aliases.add(text)
        records.append(
            {
                "label": label,
                "people": people,
                "aliases": {
                    alias.lower()
                    for alias in aliases
                    if len(alias.strip()) >= 4 and not alias.startswith("http")
                },
            }
        )
    return records


def _nearest_svg_source_records(
    records: list[dict[str, Any]],
    lowered_lines: list[str],
    person_line_index: int,
) -> list[dict[str, Any]]:
    nearest_distance: int | None = None
    nearest: list[dict[str, Any]] = []
    for record in records:
        distances: list[int] = []
        label = str(record["label"]).lower()
        for index, line in enumerate(lowered_lines):
            if _is_svg_footer_source_line(line):
                continue
            if label and label in line:
                distances.append(abs(index - person_line_index))
        if not distances:
            continue
        distance = min(distances)
        if distance > 1:
            continue
        if nearest_distance is None or distance < nearest_distance:
            nearest_distance = distance
            nearest = [record]
        elif distance == nearest_distance:
            nearest.append(record)
    return nearest


def _is_svg_footer_source_line(line: str) -> bool:
    return (
        line.count("src-") > 1
        or "source transcript" in line
        or "openreview paper" in line
        or "paper:" in line
        or "poster-" in line
        or "seg:" in line
    )


def _svg_fragment_lines(visible_text: str) -> list[str]:
    fragments: list[str] = []
    for line in visible_text.splitlines():
        cleaned = " ".join(line.split()).strip()
        if len(cleaned) < 16:
            continue
        tail_word = cleaned.rsplit(" ", 1)[-1]
        if not (cleaned.endswith("-") or _is_fragment_end_word(tail_word)):
            continue
        # A wrapped proper-noun title (mostly Title Case, e.g. "Strategic Scaling
        # of") is not a sentence fragment; only flag lowercase sentence-flow lines.
        words = cleaned.split()
        capitalized = sum(1 for word in words if word[:1].isupper())
        if not cleaned.endswith("-") and len(words) >= 2 and capitalized >= max(2, len(words) - 1):
            continue
        fragments.append(cleaned)
    return fragments


def _render_visual(slide: dict[str, Any], *, deck_dir: Path, workspace: Path) -> str:
    refs = [str(ref) for ref in slide.get("visual_asset_refs", []) if str(ref).strip()]
    if not refs:
        return (
            '<svg class="slide-visual" data-qc-role="decorative-art" viewBox="0 0 480 280" '
            'role="img" aria-label="Structured visual">'
            '<rect width="480" height="280" rx="18" fill="#158158"/>'
            '<circle cx="120" cy="138" r="58" fill="#a8d8ea"/>'
            '<path d="M198 140h160" stroke="#f8fafc" stroke-width="16" stroke-linecap="round"/>'
            '<text x="42" y="248" font-size="24" fill="#f8fafc">Structured visual</text>'
            "</svg>"
        )
    src = _asset_src(refs[0], deck_dir=deck_dir, workspace=workspace)
    alt = html.escape(_slide_title(slide, 1), quote=True)
    return f'<img class="slide-visual proof-visual" data-qc-role="figure" src="{html.escape(src, quote=True)}" alt="{alt}" loading="eager">'


def _brand_asset_refs(deck_dir: Path) -> list[dict[str, str]]:
    role_paths = dict(BRAND_ASSETS)
    role_paths["section_bg"] = BRAND_ASSETS["decorative_art"]
    refs: list[dict[str, str]] = []
    for role in ("logo", "decorative_art", "bug", "section_bg"):
        path = role_paths[role]
        refs.append(
            {
                "role": role,
                "path": _html_relative_src(path, deck_dir),
                "sha256": _sha256_file(path) if path.is_file() else "",
            }
        )
    return refs


def _mapping_payload(payload: dict[str, Any], *, render_spec: Path) -> dict[str, Any]:
    slides = [slide for slide in payload.get("slides", []) if isinstance(slide, dict)]
    content_rows: list[dict[str, Any]] = []
    mapping: dict[str, Any] = {
        "schema_version": 1,
        "render_spec": render_spec.name,
    }
    for index, slide in enumerate(slides, start=1):
        source_ids = [
            str(chip.get("source_id") or chip.get("label"))
            for chip in slide.get("source_chips", [])
            if isinstance(chip, dict) and (chip.get("source_id") or chip.get("label"))
        ]
        row = {
            "slide_id": str(slide.get("slide_id") or f"slide-{index:03d}"),
            "slide_type": "title" if index == 1 else "content",
            "html_section_index": index - 1,
            "frame_order": index,
            "source_ids": source_ids,
            "visual_asset_refs": [
                str(ref) for ref in slide.get("visual_asset_refs", []) if str(ref).strip()
            ],
            "delivery_state_id": slide.get("state_id"),
        }
        content_rows.append(row)
        mapping[str(index)] = index - 1
    mapping["slides"] = content_rows
    return mapping


def _visible_text(value: object) -> str:
    """Text as a slide should show it, whatever the render spec holds.

    A render spec built before the cleaners were fixed can still carry a Markdown link
    or a leaked LaTeX macro name in its bookend rows (ICML 2026 decks 01 and 02), and
    re-running the chain that wrote them would cost far more than repairing them here.
    """

    return repair_leaked_macro_names(strip_markdown(str(value or "")))


def _render_link_list_items(
    items: list[object],
    *,
    link_resolver: SourceLinkResolver | None = None,
) -> str:
    rows: list[str] = []
    current_arc = None
    for item in items or []:
        if not isinstance(item, dict):
            continue
        # Group the list into narrative arcs when items carry an `arc` label, so a
        # long index reads as a guided itinerary instead of a flat directory. A
        # blank/absent arc falls back to the original ungrouped list.
        arc = html.escape(str(item.get("arc") or "")).strip()
        if arc and arc != current_arc:
            rows.append(f'<li class="li-arc" role="presentation">{arc}</li>')
            current_arc = arc
        name = html.escape(_visible_text(item.get("name") or item.get("title"))).strip()
        detail = html.escape(_visible_text(item.get("detail") or item.get("note"))).strip()
        url = _safe_review_href(item.get("url")) or (
            link_resolver.resolve_item(item) if link_resolver is not None else ""
        )
        label = (
            f'<a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener">{name}</a>'
            if url and name
            else name
        )
        detail_html = f'<span class="li-detail">{detail}</span>' if detail else ""
        rows.append(f'<li><span class="li-name">{label}</span>{detail_html}</li>')
    return "\n".join(rows)


# Estimated row height (see _bookend_column_height) past which the left events
# column runs into the source chips beneath it. Measured on ICML 2026: deck 02's
# column (390) cleared QC; deck 04's (469) put its last author over the chips.
# The papers column sits to the right of the chips and may run taller.
BOOKEND_COLUMN_HEIGHT_PX = 400


def _bookend_column_height(items: list[object]) -> float:
    """Rough rendered height (px) of one events/resources column at the default
    type size: names wrap near 47 characters at 20px bold, details near 66 at 16px."""

    height = 0.0
    for item in items or []:
        if not isinstance(item, dict):
            continue
        name = _visible_text(item.get("name") or item.get("title")).strip()
        detail = _visible_text(item.get("detail") or item.get("note")).strip()
        height += 23.0 * max(1, -(-len(name) // 47)) + 14.0
        if detail:
            height += 3.0 + 19.2 * -(-len(detail) // 66)
    return height


def _split_arcs_two_cols(items: list[object]) -> tuple[list[object], list[object]]:
    """Split arc-grouped index items into two balanced columns without breaking an
    arc across columns (first half of the arcs left, the rest right)."""
    groups: list[list[object]] = []
    for it in items:
        arc = str(it.get("arc") or "") if isinstance(it, dict) else ""
        if not groups or (isinstance(groups[-1][0], dict) and str(groups[-1][0].get("arc") or "") != arc):
            groups.append([])
        groups[-1].append(it)
    half = (len(groups) + 1) // 2
    left = [it for g in groups[:half] for it in g]
    right = [it for g in groups[half:] for it in g]
    return left, right


def _render_events_section(
    slide: dict[str, Any],
    *,
    slide_id: str,
    index: int,
    bug_src: str,
    number_html: str,
    deck_dir: Path,
    workspace: Path,
    link_resolver: SourceLinkResolver | None = None,
) -> str:
    heading = html.escape(_slide_title(slide, index) or "What this talk draws on")
    event_items = slide.get("events", []) or []
    paper_items = slide.get("papers", []) or []
    prepared = html.escape(str(slide.get("prepared_by") or "Prepared by Nikolaos Vasiloglou"))
    events_col_title = html.escape(
        str(
            slide.get("events_title")
            or f"Relevant events at {conference_name(workspace, default='the conference')}"
        )
    )
    papers_col_title = html.escape(str(slide.get("papers_title") or "Works featured in this talk"))
    arc_grouped = any(isinstance(it, dict) and str(it.get("arc") or "").strip() for it in event_items)
    # This layout holds ~5 events + ~5 papers. An executive topic index carries
    # one row per conference topic plus a resource link per topic, which is 2-3x
    # that and renders as an unreadable, footer-overlapped slide -- so refuse
    # rather than draw it. The itinerary branch below is the layout built for
    # those counts, and it is selected by every row carrying an arc. This check
    # is deliberately independent of the plan builder's: it also catches a
    # hand-edited render_spec.json that never passed through that code.
    if not arc_grouped and (len(event_items) > BOOKEND_COLUMN_MAX_ITEMS
                            or len(paper_items) > BOOKEND_COLUMN_MAX_ITEMS):
        partially_arced = sum(
            1 for it in event_items if isinstance(it, dict) and str(it.get("arc") or "").strip()
        )
        raise ValueError(
            f"bookend slide {slide_id!r} has {len(event_items)} events and "
            f"{len(paper_items)} resources, over the {BOOKEND_COLUMN_MAX_ITEMS}-item "
            "column budget for this layout, and no narrative arcs to group them "
            f"({partially_arced} of {len(event_items)} rows carry an 'arc'). Regenerate "
            "the deck plan so every topic row carries an arc, or cut the item counts."
        )
    if arc_grouped:
        # Executive itinerary index: the topics are grouped into narrative arcs.
        # Drop the resource column AND the bottom source chips (both live on the
        # closing slide) and split the arcs across two columns so the 12 topics
        # never overflow / chop (2026-06-23 owner feedback).
        heading = html.escape(str(slide.get("events_title") or _slide_title(slide, index) or ""))
        left, right = _split_arcs_two_cols(event_items)
        columns = [
            f'<div class="list-col"><ul>{_render_link_list_items(c, link_resolver=link_resolver)}</ul></div>'
            for c in (left, right) if c
        ]
        grid_class = "two-col-list itinerary" + ("" if len(columns) > 1 else " single")
        columns_html = "\n    ".join(columns)
        chips_html = ""
        # The itinerary index carries no source chips BY DESIGN -- its rows are
        # this conference's own topics, and the cited sources live on the
        # closing resources slide. Declared explicitly so rendered_qc can tell
        # a deliberate omission from chips that failed to render.
        chips_attr = ' data-qc-chips="not-applicable"'
    else:
        # Render only the columns that actually have content. A topic whose sources are
        # all papers (no talks/keynotes) must not show an empty "Relevant events" column.
        columns = []
        if event_items:
            columns.append(
                f'<div class="list-col"><h3>{events_col_title}</h3>'
                f"<ul>{_render_link_list_items(event_items, link_resolver=link_resolver)}</ul></div>"
            )
        if paper_items or not columns:
            columns.append(
                f'<div class="list-col"><h3>{papers_col_title}</h3>'
                f"<ul>{_render_link_list_items(paper_items, link_resolver=link_resolver)}</ul></div>"
            )
        grid_class = "two-col-list" if len(columns) > 1 else "two-col-list single"
        columns_html = "\n    ".join(columns)
        chips_attr = ""
        chips_html = _render_source_chips(
            slide.get("source_chips", []),
            deck_dir=deck_dir,
            workspace=workspace,
            link_resolver=link_resolver,
        )
    compact = (
        " compact"
        if not arc_grouped
        and _bookend_column_height(event_items) > BOOKEND_COLUMN_HEIGHT_PX
        else ""
    )
    return f"""<section class="slide-events{compact}" data-slide-id="{slide_id}" data-slide-type="events" data-markdown-slide="{index}"{chips_attr}>
  <h2>{heading}</h2>
  <div class="{grid_class}">
    {columns_html}
  </div>
  <div class="bookend-reference-chips">{chips_html}</div>
  <p class="prepared-by">{prepared}</p>
  <img class="content-brand-bug" data-qc-role="content-brand-bug" src="{bug_src}" alt="" aria-hidden="true">
  {number_html}
</section>"""


def _render_resources_section(
    slide: dict[str, Any],
    *,
    slide_id: str,
    index: int,
    bug_src: str,
    number_html: str,
    deck_dir: Path,
    workspace: Path,
    link_resolver: SourceLinkResolver | None = None,
) -> str:
    heading = html.escape(_slide_title(slide, index) or "Sources and further reading")
    rows: list[str] = []
    for item in slide.get("resources", []) or []:
        if not isinstance(item, dict):
            continue
        title = html.escape(_visible_text(item.get("title"))).strip()
        note = html.escape(_visible_text(item.get("note"))).strip()
        url = _safe_review_href(item.get("url")) or (
            link_resolver.resolve_item(item) if link_resolver is not None else ""
        )
        link = (
            f'<a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener">{title}</a>'
            if url and title
            else title
        )
        rows.append(f'<li>{link}{(" &mdash; " + note) if note else ""}</li>')
    resources = "\n".join(rows)
    analysis_url = str(slide.get("analysis_url") or "").strip()
    analysis = (
        f'<p class="analysis-link">{html.escape(str(slide.get("analysis_note") or "These are some of the sources featured in this talk. For the full set of papers and talks, see the"))} '
        f'<a href="{html.escape(analysis_url, quote=True)}">comprehensive analysis report</a>.</p>'
        if analysis_url
        else ""
    )
    chips_html = _render_source_chips(
        slide.get("source_chips", []),
        deck_dir=deck_dir,
        workspace=workspace,
        link_resolver=link_resolver,
    )
    return f"""<section class="slide-resources" data-slide-id="{slide_id}" data-slide-type="resources" data-markdown-slide="{index}">
  <h2>{heading}</h2>
  <ol class="resources-list">{resources}</ol>
  <div class="bookend-reference-chips">{chips_html}</div>
  {analysis}
  <img class="content-brand-bug" data-qc-role="content-brand-bug" src="{bug_src}" alt="" aria-hidden="true">
  {number_html}
</section>"""


def _title_eyebrow(payload: dict[str, Any], *, conference_label: str = "") -> str:
    """Conference + topic eyebrow derived from the render spec — never a
    hardcoded topic string (the canary-era template printed 'ICLR 2026 Topic 4'
    on every deck). ``conference_label`` is the workspace-resolved conference
    name supplied by the caller when the render spec omits it."""

    label = str(
        payload.get("conference_label") or conference_label or "the conference"
    ).strip()
    presentation_id = str(payload.get("presentation_id") or "").strip()
    match = re.match(r"^0*(\d+)_", presentation_id)
    if match:
        return f"{label} Topic {int(match.group(1))}"
    return label


def render_spec_to_html(
    render_spec: Path,
    *,
    output_html: Path,
    slide_mapping: Path,
    workspace: Path | None = None,
    require_brand_assets: bool = True,
) -> Path:
    """Render structured ``render_spec.json`` into inspectable branded HTML."""

    render_spec = render_spec.expanduser().resolve()
    output_html = output_html.expanduser().resolve()
    slide_mapping = slide_mapping.expanduser().resolve()
    workspace = workspace.expanduser().resolve() if workspace is not None else _infer_workspace(output_html)
    deck_dir = output_html.parent
    payload = _load_json_object(render_spec)
    slides = [slide for slide in payload.get("slides", []) if isinstance(slide, dict)]
    if not slides:
        raise ValueError(f"{render_spec} must contain at least one slide")
    legibility_findings = _visual_legibility_findings(slides, deck_dir=deck_dir, workspace=workspace)
    if legibility_findings:
        raise ValueError(
            "visual legibility gate failed before render export: "
            + "; ".join(legibility_findings[:12])
        )
    missing_brand_assets = [str(path) for path in BRAND_ASSETS.values() if not path.is_file()]
    if require_brand_assets and missing_brand_assets:
        raise FileNotFoundError(
            "Missing RelationalAI brand assets required for structured render: "
            + ", ".join(missing_brand_assets)
        )

    brand_hash = brand_assets_sha256()
    brand_refs = _brand_asset_refs(deck_dir)
    link_resolver = SourceLinkResolver.from_workspace(workspace)
    brand_refs_by_role = {ref["role"]: ref for ref in brand_refs}
    css_vars = "\n      ".join(f"{key}: {value};" for key, value in CSS_VARIABLES.items())
    brand_contract = {
        "reference_name": BRAND_REFERENCE_NAME,
        "brand_assets_sha256": brand_hash,
        "asset_refs": brand_refs,
        "renderer_geometry": RENDERER_GEOMETRY,
        "css_variables": CSS_VARIABLES,
    }
    resolved_conference_label = conference_name(workspace, default="the conference")
    title = str(
        payload.get("title")
        or _slide_title(slides[0], 1)
        or resolved_conference_label
    ).strip()
    report_url = str(payload.get("analysis_report_url") or "").strip()
    author = html.escape(str(payload.get("author") or "Nikolaos Vasiloglou"))
    logo_src = html.escape(brand_refs_by_role["logo"]["path"], quote=True)
    decorative_src = html.escape(brand_refs_by_role["decorative_art"]["path"], quote=True)
    bug_src = html.escape(brand_refs_by_role["bug"]["path"], quote=True)
    sections: list[str] = []
    total = len(slides)
    for index, slide in enumerate(slides, start=1):
        slide_id = html.escape(str(slide.get("slide_id") or f"slide-{index:03d}"), quote=True)
        heading = html.escape(_slide_title(slide, index))
        number_html = f'<div class="slide-number">{index:02d} / {total:02d}</div>'
        slide_kind = str(slide.get("slide_kind") or "").strip().lower()
        if index == 1:
            eyebrow = html.escape(
                _title_eyebrow(payload, conference_label=resolved_conference_label)
            )
            statement = _compact_text(slide.get("title_statement") or "", limit=120)
            title_statement_html = (
                f'<p class="title-statement">{html.escape(statement)}</p>' if statement else ""
            )
            sections.append(
                f"""<section class="slide-title title-slide" data-slide-id="{slide_id}" data-slide-type="title" data-markdown-slide="{index}">
  <img class="title-logo" data-qc-role="title-logo" src="{logo_src}" alt="RelationalAI logo">
  <img class="decorative-art" data-qc-role="decorative_art" src="{decorative_src}" alt="" aria-hidden="true">
  <p class="eyebrow">{eyebrow}</p>
  <h1>{heading}</h1>
  {title_statement_html}
  <p class="author">Author: {author}</p>
</section>"""
            )
            continue
        if slide_kind == "events":
            sections.append(
                _render_events_section(
                    slide,
                    slide_id=slide_id,
                    index=index,
                    bug_src=bug_src,
                    number_html=number_html,
                    deck_dir=deck_dir,
                    workspace=workspace,
                    link_resolver=link_resolver,
                )
            )
            continue
        if slide_kind == "resources":
            sections.append(
                _render_resources_section(
                    slide,
                    slide_id=slide_id,
                    index=index,
                    bug_src=bug_src,
                    number_html=number_html,
                    deck_dir=deck_dir,
                    workspace=workspace,
                    link_resolver=link_resolver,
                )
            )
            continue
        lesson_class = " executive-lesson" if str(slide.get("executive_lesson_id") or "").strip() else ""
        sections.append(
            f"""<section class="slide-content-with-figure{lesson_class}" data-slide-id="{slide_id}" data-slide-type="content" data-markdown-slide="{index}">
  <img class="content-brand-bug" data-qc-role="content-brand-bug" src="{bug_src}" alt="" aria-hidden="true">
  <h2>{heading}</h2>
  <div class="slide-grid">
    <div class="slide-copy">
      {_slide_copy_html(slide, heading)}
      {_render_source_chips(slide.get("source_chips", []), deck_dir=deck_dir, workspace=workspace, link_resolver=link_resolver)}
    </div>
    <figure>
      {_render_visual(slide, deck_dir=deck_dir, workspace=workspace)}
    </figure>
  </div>
  {number_html}
</section>"""
        )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="presentation-template" content="{BRAND_REFERENCE_NAME}">
  <meta name="presentation-template-sha256" content="{brand_hash}">
  <meta name="presentation-brand-assets-sha256" content="{brand_hash}">
  {chr(10).join(f'<meta name="brand-asset-{html.escape(ref["role"], quote=True)}" content="{html.escape(ref["path"], quote=True)} sha256={html.escape(ref["sha256"], quote=True)}">' for ref in brand_refs)}
  <title>{html.escape(title)}</title>
  <style>
    @page {{ size: 1280px 720px; margin: 0; }}
    :root {{
      {css_vars}
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--navy); color: #f8fafc; font-family: Arial, sans-serif; }}
    .reveal .slides section {{
      width: 1280px; height: 720px; position: relative; overflow: hidden;
      page-break-after: always; break-after: page; padding: 44px 64px;
      background: radial-gradient(circle at 84% 8%, rgba(5,141,199,0.18), transparent 28%), var(--navy);
      color: #f8fafc;
    }}
    .slide-title {{ background: var(--navy); }}
    .title-logo {{ position: absolute; left: 77px; top: 26px; width: 309px; height: 48px; object-fit: contain; }}
    .decorative-art {{ position: absolute; right: -30px; top: -30px; width: 520px; height: 530px; opacity: 0.8; object-fit: contain; }}
    .eyebrow {{ margin: 160px 0 16px; color: #a8d8ea; font-size: 24px; }}
    h1 {{ max-width: 760px; margin: 0 0 20px; font-size: 64px; line-height: 1.03; }}
    .title-statement {{ max-width: 820px; margin: 0 0 24px; color: #a8d8ea; font-size: 27px; line-height: 1.18; font-weight: 600; }}
    .author, .final-report-link {{ font-size: 24px; color: #f8fafc; }}
    .final-report-link {{ display: inline-block; margin-top: 24px; color: #a8d8ea; }}
    /* Headings use the full inner width; branding belongs on the cover and footer. */
    h2 {{ max-width: 100%; margin: 0; font-size: 34px; line-height: 1.08; }}
    .slide-grid {{ position: absolute; left: 56px; right: 56px; top: 132px; bottom: 50px; display: grid; grid-template-columns: minmax(190px, 220px) minmax(880px, 1fr); gap: 18px; align-items: stretch; }}
    .executive-lesson .slide-grid {{ grid-template-columns: minmax(240px, 270px) minmax(830px, 1fr); }}
    /* min-width: 0 overrides the flex/grid item default of min-width: auto,
       which otherwise lets an unbreakable long token (a research-thread id
       like "icml2026-...:05_scaling_efficiency_foundations:248e4ee99a5c"
       embedded in evidence_caption/visual_takeaway text) push this column
       wider than its grid track and visually overlap the adjacent figure. */
    .slide-copy {{ min-height: 0; min-width: 0; display: flex; flex-direction: column; justify-content: flex-start; gap: 14px; overflow: hidden; }}
    .slide-copy p {{ line-height: 1.22; margin: 0; color: #f3f7fb; overflow-wrap: anywhere; }}
    .evidence-copy {{ display: flex; flex-direction: column; gap: 12px; }}
    /* Reduced from 19px/17px (default line-height 1.22 inherited from
       .slide-copy p): now that _compact_text preserves complete sentences
       instead of mid-word truncating them, captions run up to ~370 chars.
       At 19px in this ~200px column that's ~18 lines / ~400px tall, leaving
       no room for the heading/chips/takeaway in the ~538px available height
       -- caught by visual QC as text_clipped/crowded_layout. At 13-14px
       with tighter line-height even the longest observed caption fits in
       ~190px, comfortably inside the panel. */
    .evidence-caption {{ font-size: 15px; line-height: 1.15; font-weight: 700; }}
    .visual-takeaway {{ font-size: 14px; line-height: 1.15; color: #d7ecf6; }}
    .evidence-source {{ font-size: 13px; color: #a8d8ea; line-height: 1.22; }}
    .source-locator {{ display: block; color: #8fb8cc; margin-top: 4px; overflow-wrap: anywhere; }}
    figure {{ margin: 0; min-height: 0; height: 100%; display: flex; align-items: center; justify-content: center; overflow: hidden; }}
    .slide-visual {{ max-width: 100%; max-height: 100%; width: auto; height: auto; display: block; background: rgba(168, 216, 234, 0.08); border-radius: 8px; }}
    .reference-chips {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
    .reference-chips span {{ font-weight: 700; color: #a8d8ea; }}
    .source-chip {{ color: #f8fafc; border: 1px solid rgba(168,216,234,0.65); padding: 6px 8px; border-radius: 6px; text-decoration: none; font-size: 15px; max-width: 100%; overflow-wrap: anywhere; white-space: normal; line-height: 1.25; }}
    .source-chip-dual a {{ color: inherit; text-decoration: none; }}
    .source-chip-poster {{ color: #a8d8ea; font-size: 13px; margin-left: 6px; white-space: nowrap; }}
    .content-brand-bug {{ position: absolute; right: 24px; bottom: 18px; width: 54px; height: 40px; object-fit: contain; }}
    .slide-number {{ position: absolute; left: 64px; bottom: 16px; font-size: 13px; letter-spacing: 0.05em; color: #8fb8cc; }}
    .slide-events, .slide-resources {{ background: var(--navy); }}
    .two-col-list {{ position: absolute; left: 64px; right: 64px; top: 150px; bottom: 132px; display: grid; grid-template-columns: 1fr 1fr; gap: 44px; }}
    .two-col-list.single {{ left: 64px; right: 64px; grid-template-columns: minmax(0, 760px); justify-content: center; }}
    .list-col h3 {{ margin: 0 0 16px; color: #a8d8ea; font-size: 22px; }}
    .list-col ul {{ list-style: none; margin: 0; padding: 0; }}
    .list-col li {{ margin-bottom: 18px; }}
    .li-name {{ display: block; font-size: 20px; font-weight: 700; color: #f8fafc; }}
    .li-name a {{ color: #a8d8ea; text-decoration: none; }}
    .li-detail {{ display: block; font-size: 15px; color: #cfe3ef; margin-top: 2px; }}
    .slide-events .two-col-list {{ top: 150px; bottom: 150px; gap: 56px; }}
    .slide-events .list-col h3 {{ margin-bottom: 14px; font-size: 24px; }}
    .slide-events .list-col li {{ margin-bottom: 14px; }}
    .slide-events .li-name {{ font-size: 20px; line-height: 1.15; }}
    .slide-events .li-detail {{ font-size: 16px; line-height: 1.2; margin-top: 3px; }}
    .slide-events.compact .list-col h3 {{ font-size: 21px; margin-bottom: 10px; }}
    .slide-events.compact .list-col li {{ margin-bottom: 9px; }}
    .slide-events.compact .li-name {{ font-size: 17px; line-height: 1.15; }}
    .slide-events.compact .li-detail {{ font-size: 14px; line-height: 1.2; margin-top: 2px; }}
    .li-arc {{ list-style: none; font-size: 13px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: #6fb8d6; margin: 14px 0 8px; padding-bottom: 4px; border-bottom: 1px solid rgba(111, 184, 214, 0.32); }}
    .li-arc:first-child {{ margin-top: 0; }}
    .slide-events .li-arc {{ font-size: 12px; margin: 10px 0 6px; }}
    /* Executive itinerary index: no resource column / chips, so use the full
       height and give the two arc columns room to breathe. */
    .slide-events .two-col-list.itinerary {{ top: 128px; bottom: 64px; gap: 72px; align-content: start; }}
    .slide-events .two-col-list.itinerary .li-name {{ font-size: 19px; line-height: 1.16; }}
    .slide-events .two-col-list.itinerary .li-detail {{ font-size: 14px; line-height: 1.22; margin-top: 3px; }}
    .slide-events .two-col-list.itinerary .list-col li {{ margin-bottom: 15px; }}
    .slide-events .two-col-list.itinerary .li-arc {{ font-size: 13px; margin: 0 0 10px; }}
    .slide-events .two-col-list.itinerary .list-col + .list-col .li-arc:first-child {{ margin-top: 0; }}
    .prepared-by {{ position: absolute; left: 64px; bottom: 34px; font-size: 16px; color: #d7ecf6; font-style: italic; }}
    .resources-list {{ position: absolute; left: 64px; right: 64px; top: 124px; bottom: 124px; margin: 0; padding-left: 28px; font-size: 14px; line-height: 1.22; column-count: 2; column-gap: 48px; }}
    .resources-list li {{ margin-bottom: 6px; break-inside: avoid; }}
    .resources-list a {{ color: #a8d8ea; text-decoration: none; }}
    .bookend-reference-chips {{ position: absolute; left: 64px; right: 190px; bottom: 88px; }}
    .bookend-reference-chips .reference-chips {{ gap: 8px; }}
    .bookend-reference-chips .reference-chips > span:first-child {{ font-size: 13px; }}
    .bookend-reference-chips .source-chip {{ font-size: 13px; padding: 4px 6px; }}
    .analysis-link {{ position: absolute; left: 64px; right: 64px; bottom: 44px; font-size: 17px; color: #d7ecf6; }}
    .analysis-link a {{ color: #a8d8ea; font-weight: 700; }}
  </style>
  <script type="application/json" id="brand-template-lock">{json.dumps(brand_contract, sort_keys=True)}</script>
</head>
<body class="structured-render relationalai-neurips2025-reference">
<div class="reveal"><div class="slides">
{chr(10).join(sections)}
</div></div>
</body>
</html>
"""
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html_text, encoding="utf-8")
    write_json_atomic(slide_mapping, _mapping_payload(payload, render_spec=render_spec))
    return output_html


def export_pdf_with_playwright(*, input_html: Path, output_pdf: Path, refuse_local: bool = False) -> list[str]:
    """Export HTML to PDF using Playwright Chromium.

    With ``refuse_local``, the page loads nothing but itself, ``https:``, ``data:``,
    ``blob:`` and ``about:`` URLs: any other load, however the page asks for it (a script, a
    CSS escape, an iframe's srcdoc), is refused, and the refused URLs are returned."""

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "playwright is required for PDF export. Install project dependencies "
            "and run `playwright install chromium`."
        ) from exc
    if not input_html.is_file():
        raise FileNotFoundError(f"input HTML does not exist: {input_html}")
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        refused: list[str] = []
        try:
            page = browser.new_page(viewport={"width": DEFAULT_VIEWPORT[0], "height": DEFAULT_VIEWPORT[1]})
            target_url = input_html.resolve().as_uri()
            separator = "&" if "?" in target_url else "?"
            if refuse_local:
                def _guard(route: Any) -> None:
                    url = route.request.url
                    if url.split("?", 1)[0] == target_url or url.lower().startswith(("https:", "data:", "blob:", "about:")):
                        route.continue_()
                    else:
                        refused.append(url)
                        route.abort()

                page.route("**/*", _guard)
            page.goto(f"{target_url}{separator}print-pdf", wait_until="networkidle")
            page.emulate_media(media="print")
            page.pdf(
                path=str(output_pdf),
                print_background=True,
                prefer_css_page_size=True,
                width=f"{DEFAULT_VIEWPORT[0]}px",
                height=f"{DEFAULT_VIEWPORT[1]}px",
            )
        finally:
            browser.close()
    return sorted(set(refused))


def _frame_entries_from_mapping(slide_mapping: Path) -> list[dict[str, Any]]:
    payload = _load_json_object(slide_mapping)
    entries: list[dict[str, Any]] = []
    title = payload.get("title_slide")
    if isinstance(title, dict):
        entries.append(title)
    for slide in payload.get("slides", []):
        if isinstance(slide, dict):
            entries.append(slide)
    if entries:
        return entries
    for key in sorted(
        (key for key in payload if str(key).isdigit()),
        key=lambda item: int(str(item)),
    ):
        entries.append(
            {
                "slide_id": f"slide-{int(key):03d}",
                "html_section_index": payload[key],
                "frame_order": int(key),
            }
        )
    return entries


def export_slide_frames_with_playwright(
    *,
    html_path: Path,
    slide_mapping: Path,
    frames_dir: Path,
) -> list[dict[str, Any]]:
    """Export one PNG frame per mapped rendered slide using Playwright Chromium."""

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "playwright is required for slide frame export. Install project dependencies "
            "and run `playwright install chromium`."
        ) from exc
    entries = _frame_entries_from_mapping(slide_mapping)
    if not entries:
        raise ValueError(f"{slide_mapping} does not contain mapped slides")
    frames_dir.mkdir(parents=True, exist_ok=True)
    for stale in frames_dir.glob("slide_*.png"):
        if stale.is_file():
            stale.unlink()
    exported: list[dict[str, Any]] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": DEFAULT_VIEWPORT[0], "height": DEFAULT_VIEWPORT[1]})
            page.goto(f"{html_path.resolve().as_uri()}?hidecontrols", wait_until="networkidle")
            for output_index, entry in enumerate(entries, start=1):
                section_index = int(entry.get("html_section_index") or 0)
                page.evaluate(_JS_SHOW_SLIDE_FOR_EXPORT, section_index)
                frame_path = frames_dir / f"slide_{output_index:03d}.png"
                page.screenshot(path=str(frame_path), full_page=False)
                exported.append(
                    {
                        "slide_id": str(entry.get("slide_id") or f"slide-{output_index:03d}"),
                        "index": output_index,
                        "frame_order": int(entry.get("frame_order") or output_index),
                        "frame_path": frame_path,
                    }
                )
        finally:
            browser.close()
    return exported


def _frame_manifest_payload(
    *,
    render_spec: Path,
    slide_mapping: Path,
    frames: list[dict[str, Any]],
    frame_manifest: Path,
    workspace: Path,
) -> dict[str, Any]:
    from PIL import Image

    deck_dir = frame_manifest.parent
    frame_rows: list[dict[str, Any]] = []
    for frame in frames:
        frame_path = Path(frame["frame_path"]).resolve()
        with Image.open(frame_path) as image:
            width, height = image.size
        frame_rows.append(
            {
                "slide_id": str(frame.get("slide_id")),
                "index": int(frame.get("index") or len(frame_rows) + 1),
                "frame_order": int(frame.get("frame_order") or len(frame_rows) + 1),
                "path": _relative(frame_path, deck_dir),
                "frame_path": _relative(frame_path, deck_dir),
                "sha256": _sha256_file(frame_path),
                "byte_count": frame_path.stat().st_size,
                "dimensions": {"width": width, "height": height},
                "render_input_hashes": {
                    "render_spec.json": _sha256_file(render_spec),
                    "presentation_slide_mapping.json": _sha256_file(slide_mapping),
                },
            }
        )
    return {
        "schema_version": 1,
        "generated_at": _isoformat_z(),
        "render_spec_path": _relative(render_spec, workspace),
        "slide_mapping_path": _relative(slide_mapping, workspace),
        "frames": frame_rows,
    }


def _make_contact_sheet(frame_paths: list[Path], output_path: Path) -> bool:
    from PIL import Image, ImageDraw

    existing = [path for path in frame_paths if path.is_file()]
    if not existing:
        return False
    thumb_width, thumb_height = 320, 180
    columns = min(4, len(existing))
    rows = (len(existing) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_width, rows * (thumb_height + 24)), "#111827")
    draw = ImageDraw.Draw(sheet)
    for index, frame_path in enumerate(existing):
        with Image.open(frame_path) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((thumb_width, thumb_height))
            x = (index % columns) * thumb_width
            y = (index // columns) * (thumb_height + 24)
            sheet.paste(thumb, (x, y))
            draw.text((x + 8, y + thumb_height + 4), frame_path.stem, fill="#f8fafc")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return output_path.is_file()


def export_render_spec_artifacts(
    *,
    render_spec: Path,
    output_html: Path,
    output_pdf: Path,
    slide_mapping: Path,
    frame_manifest: Path,
    frames_dir: Path,
    contact_sheet: Path,
    receipt_output: Path,
    workspace: Path | None = None,
    command: list[str] | None = None,
) -> dict[str, Any]:
    """Render HTML/PDF/PNG frames and write hash-bound export evidence."""

    render_spec = render_spec.expanduser().resolve()
    output_html = output_html.expanduser().resolve()
    output_pdf = output_pdf.expanduser().resolve()
    slide_mapping = slide_mapping.expanduser().resolve()
    frame_manifest = frame_manifest.expanduser().resolve()
    frames_dir = frames_dir.expanduser().resolve()
    contact_sheet = contact_sheet.expanduser().resolve()
    receipt_output = receipt_output.expanduser().resolve()
    workspace = workspace.expanduser().resolve() if workspace is not None else _infer_workspace(output_html)
    deck_dir = output_html.parent

    render_spec_to_html(render_spec, output_html=output_html, slide_mapping=slide_mapping, workspace=workspace)
    export_pdf_with_playwright(input_html=output_html, output_pdf=output_pdf)
    frames = export_slide_frames_with_playwright(
        html_path=output_html,
        slide_mapping=slide_mapping,
        frames_dir=frames_dir,
    )
    manifest_payload = _frame_manifest_payload(
        render_spec=render_spec,
        slide_mapping=slide_mapping,
        frames=frames,
        frame_manifest=frame_manifest,
        workspace=workspace,
    )
    write_json_atomic(frame_manifest, manifest_payload)
    frame_paths = [Path(frame["frame_path"]).resolve() for frame in frames]
    contact_sheet_created = _make_contact_sheet(frame_paths, contact_sheet)
    legacy_paths = [
        _relative(path, deck_dir)
        for path in collect_legacy_markdown_paths(deck_dir)
    ]
    payload = _load_json_object(render_spec)
    input_hashes = dict(payload.get("input_hashes") if isinstance(payload.get("input_hashes"), dict) else {})
    input_hashes["render_spec.json"] = _sha256_file(render_spec)
    input_hashes["presentation.html"] = _sha256_file(output_html)
    input_hashes["presentation.pdf"] = _sha256_file(output_pdf)
    input_hashes["slide_frame_manifest.json"] = _sha256_file(frame_manifest)
    if contact_sheet_created:
        input_hashes["render_contact_sheet.png"] = _sha256_file(contact_sheet)
    receipt = {
        "schema_version": 1,
        "status": "pass" if contact_sheet_created and not legacy_paths else "fail",
        "generated_at": _isoformat_z(),
        "render_spec_path": _relative(render_spec, workspace),
        "render_spec_sha256": _sha256_file(render_spec),
        "input_hashes": input_hashes,
        "export_commands": [
            {
                "command": command or [],
                "engine": "Playwright Chromium",
                "outputs": [
                    _relative(output_html, workspace),
                    _relative(output_pdf, workspace),
                    _relative(slide_mapping, workspace),
                    _relative(frame_manifest, workspace),
                ],
            }
        ],
        "slide_count": len(frames),
        "html_path": _relative(output_html, workspace),
        "pdf_path": _relative(output_pdf, workspace),
        "slide_mapping_path": _relative(slide_mapping, workspace),
        "frame_manifest": {
            "path": _relative(frame_manifest, workspace),
            "sha256": _sha256_file(frame_manifest),
        },
        "frame_manifest_sha256": _sha256_file(frame_manifest),
        "contact_sheet_path": str(contact_sheet),
        "contact_sheet_sha256": _sha256_file(contact_sheet) if contact_sheet_created else None,
        "legacy_markdown_paths": legacy_paths,
    }
    write_json_atomic(receipt_output, receipt)
    return receipt


def _write_slides_concat(path: Path, segments: list[dict[str, Any]]) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = 0.0
    lines: list[str] = []
    for index, segment in enumerate(segments, start=1):
        duration = _segment_duration(segment)
        total += duration
        lines.extend([f"file 'structured-slide-{index:03d}.png'", f"duration {duration:.3f}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return total


_JS_SHOW_SLIDE_FOR_EXPORT = """
(index) => {
  const sections = Array.from(document.querySelectorAll('section'));
  document.body.style.margin = '0';
  document.body.style.overflow = 'hidden';
  sections.forEach((section, current) => {
    section.style.display = current === index ? 'block' : 'none';
    section.style.visibility = current === index ? 'visible' : 'hidden';
    section.style.opacity = current === index ? '1' : '0';
    section.style.position = current === index ? 'relative' : 'absolute';
    section.style.left = '0';
    section.style.top = '0';
    section.style.width = '1280px';
    section.style.height = '720px';
    section.style.transform = 'none';
  });
}
"""


def build_structured_delivery_video(
    *,
    delivery_contract: Path,
    slides_html: Path,
    slide_mapping: Path | None,
    build_dir: Path,
    output_video: Path,
    style: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Create short, valid media artifacts from a delivery contract.

    The structured canary path validates render/delivery synchronization without
    routing through the legacy provider-heavy narration builders. It still writes
    real MP4/M4A files so ffprobe/ffmpeg-based gates can inspect duration and
    tail-frame evidence.
    """

    payload = _load_json_object(delivery_contract)
    segments = [
        segment
        for segment in payload.get("segments", [])
        if isinstance(segment, dict)
    ]
    if not segments:
        raise ValueError(f"{delivery_contract} must contain at least one segment")
    if not slides_html.exists():
        raise FileNotFoundError(f"slides HTML not found: {slides_html}")
    if slide_mapping is not None and not slide_mapping.exists():
        raise FileNotFoundError(f"slide mapping not found: {slide_mapping}")

    total_duration = sum(_segment_duration(segment) for segment in segments)
    total_duration = max(total_duration, 1.0)
    plan = {
        "builder": "conference_pipeline.presentation.structured_video",
        "delivery_contract": str(delivery_contract),
        "dry_run": dry_run,
        "output": str(output_video),
        "segment_count": len(segments),
        "slide_mapping": str(slide_mapping) if slide_mapping else None,
        "slides_html": str(slides_html),
        "style": style,
        "total_duration_seconds": round(total_duration, 3),
        "provider_requirements": ["tool:ffmpeg", "tool:ffprobe"],
    }
    if dry_run:
        return plan

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for structured delivery video generation")

    build_dir.mkdir(parents=True, exist_ok=True)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    script_json = build_dir / "script.json"
    slides_concat = build_dir / "slides_concat.txt"
    narration_track = build_dir / "narration_track.m4a"
    metadata = build_dir / "metadata.json"

    _write_json(
        script_json,
        {
            "schema_version": 1,
            "source_delivery_contract": str(delivery_contract),
            "style": style,
            "segments": segments,
        },
    )
    slides_duration = _write_slides_concat(slides_concat, segments)

    audio_cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:duration={total_duration:.3f}",
        "-af",
        "volume=0.02",
        "-c:a",
        "aac",
        str(narration_track),
    ]
    _run(audio_cmd)

    color = "0xF4F7FB" if style == "single_speaker" else "0xEFFAF4"
    video_cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=c={color}:s=1280x720:r=30:d={total_duration:.3f}",
        "-i",
        str(narration_track),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        "-movflags",
        "+faststart",
        str(output_video),
    ]
    _run(video_cmd)
    _write_json(
        metadata,
        {
            **plan,
            "dry_run": False,
            "audio_duration_seconds": round(total_duration, 3),
            "slides_duration_seconds": round(slides_duration, 3),
            "video_duration_seconds": round(total_duration, 3),
        },
    )
    return {**plan, "dry_run": False, "metadata": str(metadata)}
