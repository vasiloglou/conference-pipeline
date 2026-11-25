"""Ground presentation figures on real paper figures.

For a slide bound to an OpenReview paper this module:

1. resolves the OpenReview note (title, authors, license) — the license field
   is the machine-verifiable gate for adapting the paper's figure;
2. ensures the paper PDF is available locally (cache hit or direct download);
3. extracts candidate figure crops with the existing PyMuPDF extractor
   (crops are MODEL INPUT ONLY — they are never published);
4. pools candidates from ALL resolvable bound papers and ranks the pool
   against the slide claim with ONE receipted Claude vision call that judges
   provenance (does THIS paper make THIS claim) and returns a ``claim_match``
   confidence; below-threshold confidence yields a ``GroundingSkip`` signal
   (the redraw tier is skipped) instead of a wrong-figure grounding block
   (CALIB-04).
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz

from conference_pipeline.figure_extractor import extract_figures_from_paper
from conference_pipeline.provider_receipts import anthropic_messages_create

OPENREVIEW_NOTES_API = "https://api2.openreview.net/notes"
OPENREVIEW_PDF_URL = "https://openreview.net/pdf?id={note_id}"
ADAPTABLE_LICENSES = frozenset({"CC BY 4.0", "CC BY-SA 4.0", "CC0 1.0"})
# Per-paper candidate cap applied while pooling in ground_slide_figure.
RANKER_MAX_CANDIDATES = 8
# Overall pooled-candidate cap for the single ranking call (RESEARCH Open
# Question 2: cap per paper 8, overall 16, record truncation).
RANKER_MAX_POOLED_CANDIDATES = 16
RANKER_IMAGE_MAX_DIM = 1024
# Prior confidence threshold for the provenance claim_match gate (CALIB-04).
# Planner-chosen prior (RESEARCH Pattern 4 recommendation); recorded in every
# receipt so Phase 02 can calibrate it empirically. Below-threshold grounding
# skips the redraw tier instead of redrawing the wrong paper's figure.
CLAIM_MATCH_THRESHOLD = 0.7


@dataclass
class GroundingResult:
    openreview_id: str
    paper_title: str
    authors: list[str]
    license: str
    license_adaptable: bool
    page: int
    bbox: list[float]
    caption: str
    crop_path: str
    crop_sha256: str
    figure_kind: str
    ranker_rationale: str
    ranker_receipt_path: str | None
    candidate_count: int
    grounding_mode: str = "vision_redraw"
    warnings: list[str] = field(default_factory=list)
    claim_match: float | None = None
    claim_match_threshold: float | None = None
    pooled_notes: list[str] = field(default_factory=list)
    pool_candidate_count: int = 0
    pool_truncated: bool = False
    # Conference display name (e.g. "ICML 2026") for the attribution caption.
    # Empty when the caller cannot resolve a workspace config; the attribution
    # line then omits the conference token rather than printing a wrong one.
    conference_label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "openreview_id": self.openreview_id,
            "paper_title": self.paper_title,
            "authors": self.authors,
            "license": self.license,
            "license_adaptable": self.license_adaptable,
            "page": self.page,
            "bbox": self.bbox,
            "caption": self.caption,
            "crop_path": self.crop_path,
            "crop_sha256": self.crop_sha256,
            "figure_kind": self.figure_kind,
            "ranker_rationale": self.ranker_rationale,
            "ranker_receipt_path": self.ranker_receipt_path,
            "candidate_count": self.candidate_count,
            "grounding_mode": self.grounding_mode,
            "warnings": self.warnings,
            "claim_match": self.claim_match,
            "claim_match_threshold": self.claim_match_threshold,
            "pooled_notes": self.pooled_notes,
            "pool_candidate_count": self.pool_candidate_count,
            "pool_truncated": self.pool_truncated,
            "conference_label": self.conference_label,
        }

    def attribution_line(self) -> str:
        first_author = self.authors[0] if self.authors else "the authors"
        suffix = " et al." if len(self.authors) > 1 else ""
        title = self.paper_title
        if len(title) > 60:
            title = title[:57].rstrip(" ,;:") + "..."
        conference = str(self.conference_label or "").strip()
        # Keep the license (legally load-bearing); prepend the conference name
        # only when the caller resolved it from the workspace config.
        provenance = f"{conference}, {self.license}" if conference else self.license
        return f"Adapted from {first_author}{suffix}, {title} ({provenance})"


@dataclass
class GroundingSkip:
    """Redraw-tier skip signal (CALIB-04): grounding too uncertain to redraw.

    Returned by ``ground_slide_figure`` instead of a wrong-figure
    ``GroundingResult`` when the pooled ranker rejects every candidate
    (``reason="ranker_rejected_all"``) or its provenance confidence falls
    below threshold (``reason="claim_match_below_threshold"``). The trial
    runner converts this into a skipped outcome (wired in plan 01-03).
    """

    reason: str
    claim_match: float | None
    claim_match_threshold: float
    pooled_notes: list[str]
    pool_candidate_count: int
    ranker_rationale: str
    ranker_receipt_path: str | None
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "claim_match": self.claim_match,
            "claim_match_threshold": self.claim_match_threshold,
            "pooled_notes": self.pooled_notes,
            "pool_candidate_count": self.pool_candidate_count,
            "ranker_rationale": self.ranker_rationale,
            "ranker_receipt_path": self.ranker_receipt_path,
            "warnings": self.warnings,
        }


def bare_openreview_id(source_id: str) -> str | None:
    """Extract the bare OpenReview note id from ids like paper_abstract:y4kAMUBqLq."""

    value = str(source_id or "").strip()
    if not value:
        return None
    if ":" in value:
        family, _, bare = value.partition(":")
        if family not in {"paper_abstract", "paper", "paper_text", "openreview"}:
            return None
        value = bare
    value = value.strip()
    # OpenReview note ids are short base64-ish tokens; reject obvious hashes/paths.
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,12}", value):
        return None
    return value


def resolve_openreview_note(note_id: str, *, timeout: float = 30.0) -> dict[str, Any]:
    """Fetch note metadata (title, authors, license) from the OpenReview v2 API."""

    url = f"{OPENREVIEW_NOTES_API}?{urllib.parse.urlencode({'id': note_id})}"
    request = urllib.request.Request(url, headers={"User-Agent": "conference-pipeline-figure-grounding"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    notes = payload.get("notes") or []
    if not notes:
        raise ValueError(f"OpenReview note {note_id} not found")
    note = notes[0]
    content = note.get("content") or {}

    def _value(key: str) -> Any:
        entry = content.get(key)
        if isinstance(entry, dict):
            return entry.get("value")
        return entry

    return {
        "openreview_id": note_id,
        "title": str(_value("title") or "").strip(),
        "authors": [str(a) for a in (_value("authors") or [])],
        "license": str(note.get("license") or "").strip(),
        "venue": str(_value("venue") or "").strip(),
        "forum_url": f"https://openreview.net/forum?id={note_id}",
    }


def ensure_paper_pdf(note_id: str, cache_dir: Path, *, timeout: float = 120.0) -> Path:
    """Return a local PDF for the note, downloading from OpenReview if needed."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = cache_dir / f"{note_id}.pdf"
    if pdf_path.is_file() and pdf_path.stat().st_size > 10_000:
        return pdf_path
    url = OPENREVIEW_PDF_URL.format(note_id=note_id)
    request = urllib.request.Request(url, headers={"User-Agent": "conference-pipeline-figure-grounding"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read()
    if not data.startswith(b"%PDF"):
        raise ValueError(f"OpenReview did not return a PDF for {note_id}")
    pdf_path.write_bytes(data)
    return pdf_path


def caption_below_bbox(pdf_path: Path, page_number: int, bbox: tuple[float, float, float, float]) -> str:
    """Read the caption text directly below a figure bbox (Figure N: ...)."""

    try:
        doc = fitz.open(str(pdf_path))
        page = doc[page_number - 1]
        x0, _, x1, y1 = bbox
        zone = fitz.Rect(max(0, x0 - 10), y1, min(page.rect.x1, x1 + 10), min(page.rect.y1, y1 + 90))
        text = " ".join(page.get_text(clip=zone).split())
        doc.close()
    except Exception:
        return ""
    match = re.search(r"((?:Figure|Fig\.?|Table)\s*\d+[.:].{0,400})", text)
    return (match.group(1) if match else text)[:420]


def extract_candidate_crops(pdf_path: Path, output_dir: Path) -> list[dict[str, Any]]:
    """Extract candidate figure crops (no LLM captions; caption text via PyMuPDF)."""

    output_dir.mkdir(parents=True, exist_ok=True)
    results = extract_figures_from_paper(
        pdf_path=str(pdf_path),
        output_dir=str(output_dir),
        generate_captions=False,
        render_scale=2,
        stop_at_references=True,
        verbose=False,
    )
    candidates: list[dict[str, Any]] = []
    for item in results:
        image_path = Path(str(item.get("image_path") or ""))
        if not image_path.is_file():
            continue
        raw_bbox = item.get("bbox") or [0, 0, 0, 0]
        if isinstance(raw_bbox, dict):
            bbox = [raw_bbox.get("x0", 0), raw_bbox.get("y0", 0), raw_bbox.get("x1", 0), raw_bbox.get("y1", 0)]
        else:
            bbox = list(raw_bbox)
        page = int(item.get("page") or 0)
        candidates.append(
            {
                "page": page,
                "bbox": [float(v) for v in bbox],
                "image_path": str(image_path),
                "caption": caption_below_bbox(pdf_path, page, tuple(float(v) for v in bbox)),
            }
        )
    return candidates


def _encode_image_for_ranker(image_path: Path) -> dict[str, Any]:
    from PIL import Image
    import io

    img = Image.open(image_path).convert("RGB")
    if max(img.size) > RANKER_IMAGE_MAX_DIM:
        ratio = RANKER_IMAGE_MAX_DIM / max(img.size)
        img = img.resize((max(1, int(img.width * ratio)), max(1, int(img.height * ratio))))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    data = base64.standard_b64encode(buffer.getvalue()).decode("ascii")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": data},
    }


def rank_candidates(
    *,
    client: Any,
    caller: str,
    model: str,
    candidates: list[dict[str, Any]],
    slide_claim: str,
    slide_title: str,
    on_call: Any = None,
) -> dict[str, Any]:
    """One receipted vision call over the POOLED candidates from all bound papers.

    Each candidate dict carries ``paper_title`` and ``note_id`` (tagged during
    pooling in ``ground_slide_figure``) in addition to page/bbox/image_path/
    caption. The verdict includes ``claim_match`` — confidence in [0, 1] that
    the chosen candidate's OWN paper is the actual source of the slide claim
    (provenance, not topical similarity). Untrusted model JSON is validated:
    malformed or missing ``claim_match`` is coerced to 0.0 with a recorded
    ``claim_match_parse_warning`` (CALIB-04 / threat T-01-01).
    """

    pool_truncated = len(candidates) > RANKER_MAX_POOLED_CANDIDATES
    short_list = candidates[:RANKER_MAX_POOLED_CANDIDATES]
    content: list[dict[str, Any]] = []
    for index, candidate in enumerate(short_list):
        content.append(
            {
                "type": "text",
                "text": (
                    f"Candidate {index} (paper: {candidate['paper_title']}; "
                    f"page {candidate['page']}): {candidate['caption'] or 'no caption found'}"
                ),
            }
        )
        content.append(_encode_image_for_ranker(Path(candidate["image_path"])))
    content.append(
        {
            "type": "text",
            "text": (
                f"Slide title: {slide_title}\n"
                f"Slide claim to ground: {slide_claim}\n\n"
                "The candidate figures above come from MULTIPLE different papers; each "
                "candidate is tagged with the title of its own paper. Pick a candidate "
                "ONLY if that candidate's OWN paper actually makes the slide claim — "
                "judge provenance (does THIS paper make THIS claim), not topical overlap "
                "with axis labels, chart styling, or shared vocabulary. If the claim "
                "names a specific method, system, or benchmark that matches none of the "
                "candidate papers' titles or captions, return chosen_index -1 or a LOW "
                "claim_match. claim_match is your confidence in [0.0, 1.0] that the "
                "chosen candidate's paper is the actual source of the slide claim. "
                "Prefer method/architecture schematics and informative results figures "
                "over logos, tables of text, or decorative images. "
                "Respond with JSON only: {\"chosen_index\": <int or -1 if none usable>, "
                "\"claim_match\": <float 0.0-1.0>, "
                "\"figure_kind\": \"architecture|results_chart|mechanism|table|other\", "
                "\"rationale\": \"<one sentence naming the paper>\"}"
            ),
        }
    )
    result = anthropic_messages_create(
        client,
        caller=caller,
        model=model,
        max_tokens=400,
        messages=[{"role": "user", "content": content}],
    )
    if on_call is not None:
        on_call(result)
    text = "".join(
        block.text for block in result.response.content if getattr(block, "type", "") == "text"
    )
    receipt_path = getattr(result.receipt, "receipt_path", None)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    fallback = {"chosen_index": -1, "figure_kind": "other", "rationale": text[:200]}
    try:
        verdict = json.loads(match.group(0)) if match else dict(fallback)
    except json.JSONDecodeError:
        verdict = dict(fallback)
    # Untrusted model output (ASVS V5): claim_match must be a finite float in
    # [0, 1]; anything else coerces to 0.0 with a recorded warning so a
    # malformed response never crashes the batch.
    raw_claim_match = verdict.get("claim_match")
    try:
        claim_match = float(raw_claim_match)
        if not math.isfinite(claim_match):
            raise ValueError(f"non-finite claim_match {raw_claim_match!r}")
    except (TypeError, ValueError):
        if "claim_match" not in verdict:
            verdict["claim_match_parse_warning"] = "claim_match missing from ranker JSON"
        else:
            verdict["claim_match_parse_warning"] = (
                f"non-numeric claim_match {raw_claim_match!r} coerced to 0.0"
            )
        claim_match = 0.0
    verdict["claim_match"] = min(1.0, max(0.0, claim_match))
    if pool_truncated:
        verdict["pool_truncated"] = True
    verdict["receipt_path"] = receipt_path
    return verdict


def ground_slide_figure(
    *,
    client: Any,
    caller: str,
    ranker_model: str,
    source_ids: list[str],
    slide_claim: str,
    slide_title: str,
    grounding_root: Path,
    claim_match_threshold: float = CLAIM_MATCH_THRESHOLD,
    conference_label: str = "",
    on_call: Any = None,
) -> GroundingResult | GroundingSkip | None:
    """Full grounding flow for one slide (CALIB-04: collect-all-then-rank-once).

    Collects candidate crops from ALL resolvable bound papers into one pool,
    makes ONE pooled provenance-focused ranking call, and decides:

    - empty pool -> ``None`` (unchanged contract for "nothing usable");
    - ranker rejects every candidate -> ``GroundingSkip("ranker_rejected_all")``;
    - ``claim_match`` below threshold -> ``GroundingSkip("claim_match_below_threshold")``
      (the redraw tier is skipped instead of redrawing the wrong figure);
    - otherwise -> ``GroundingResult`` for the chosen candidate's paper.
    """

    warnings: list[str] = []
    notes_by_id: dict[str, dict[str, Any]] = {}
    pooled: list[dict[str, Any]] = []
    for source_id in source_ids:
        note_id = bare_openreview_id(source_id)
        if not note_id:
            continue
        if note_id in notes_by_id:
            continue  # same paper bound via multiple source families — pool once
        try:
            note = resolve_openreview_note(note_id)
        except Exception as exc:  # noqa: BLE001 — record and try the next source
            warnings.append(f"{source_id}: note lookup failed ({exc})")
            continue
        try:
            pdf_path = ensure_paper_pdf(note_id, grounding_root / "papers")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"{source_id}: pdf fetch failed ({exc})")
            continue
        crops_dir = grounding_root / "crops" / note_id
        try:
            candidates = extract_candidate_crops(pdf_path, crops_dir)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"{source_id}: crop extraction failed ({exc})")
            continue
        if not candidates:
            warnings.append(f"{source_id}: no figure candidates found in PDF")
            continue
        if len(candidates) > RANKER_MAX_CANDIDATES:
            warnings.append(
                f"{source_id}: capped {note_id} candidates at {RANKER_MAX_CANDIDATES} "
                f"of {len(candidates)} (page order preserved)"
            )
            candidates = candidates[:RANKER_MAX_CANDIDATES]
        notes_by_id[note_id] = note
        for candidate in candidates:
            pooled.append({**candidate, "paper_title": note["title"], "note_id": note_id})
    if not pooled:
        return None
    pooled_notes = list(notes_by_id)
    verdict = rank_candidates(
        client=client,
        caller=caller,
        model=ranker_model,
        candidates=pooled,
        slide_claim=slide_claim,
        slide_title=slide_title,
        on_call=on_call,
    )
    if "claim_match_parse_warning" in verdict:
        warnings.append(f"ranker: {verdict['claim_match_parse_warning']}")
    claim_match = float(verdict.get("claim_match") or 0.0)
    rationale = str(verdict.get("rationale") or "")
    receipt_path = verdict.get("receipt_path")
    try:
        chosen_index = int(verdict.get("chosen_index", -1))
    except (TypeError, ValueError):
        chosen_index = -1
    # The ranker only saw the first RANKER_MAX_POOLED_CANDIDATES candidates;
    # bounds-check against what it actually saw (threat T-01-01).
    ranked_count = min(len(pooled), RANKER_MAX_POOLED_CANDIDATES)
    if chosen_index < 0 or chosen_index >= ranked_count:
        warnings.append(f"ranker rejected all {ranked_count} pooled candidates")
        return GroundingSkip(
            reason="ranker_rejected_all",
            claim_match=claim_match,
            claim_match_threshold=claim_match_threshold,
            pooled_notes=pooled_notes,
            pool_candidate_count=len(pooled),
            ranker_rationale=rationale,
            ranker_receipt_path=receipt_path,
            warnings=warnings,
        )
    if claim_match < claim_match_threshold:
        return GroundingSkip(
            reason="claim_match_below_threshold",
            claim_match=claim_match,
            claim_match_threshold=claim_match_threshold,
            pooled_notes=pooled_notes,
            pool_candidate_count=len(pooled),
            ranker_rationale=rationale,
            ranker_receipt_path=receipt_path,
            warnings=warnings,
        )
    chosen = pooled[chosen_index]
    note = notes_by_id[chosen["note_id"]]
    crop_path = Path(chosen["image_path"])
    license_value = note["license"] or "unknown"
    return GroundingResult(
        openreview_id=chosen["note_id"],
        paper_title=note["title"],
        authors=note["authors"],
        license=license_value,
        license_adaptable=license_value in ADAPTABLE_LICENSES,
        page=int(chosen["page"]),
        bbox=[float(v) for v in chosen["bbox"]],
        caption=str(chosen["caption"]),
        crop_path=str(crop_path),
        crop_sha256=hashlib.sha256(crop_path.read_bytes()).hexdigest(),
        figure_kind=str(verdict.get("figure_kind") or "other"),
        ranker_rationale=rationale,
        ranker_receipt_path=receipt_path,
        candidate_count=len(pooled),
        warnings=warnings,
        claim_match=claim_match,
        claim_match_threshold=claim_match_threshold,
        pooled_notes=pooled_notes,
        pool_candidate_count=len(pooled),
        pool_truncated=bool(verdict.get("pool_truncated", False)),
        conference_label=str(conference_label or "").strip(),
    )
