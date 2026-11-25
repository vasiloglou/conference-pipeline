"""Rough-cut packet and review validation for presentation canaries."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = 1
PACKET_TYPE = "conference_pipeline.presentation.rough_cut_review"

ROUGH_CUT_MISSING_PDF_SCREENSHOTS = "rough_cut_missing_pdf_screenshots"
ROUGH_CUT_MISSING_CONTACT_SHEET = "rough_cut_missing_contact_sheet"
ROUGH_CUT_MISSING_MP4 = "rough_cut_missing_mp4"
ROUGH_CUT_VIDEO_BLACK_FRAME = "rough_cut_video_black_frame"
ROUGH_CUT_VIDEO_DESYNC = "rough_cut_video_desync"
ROUGH_CUT_MECHANICAL_DIALOGUE = "rough_cut_mechanical_dialogue"
ROUGH_CUT_VISUAL_CLAIM_MISMATCH = "rough_cut_visual_claim_mismatch"
ROUGH_CUT_SCAFFOLD_PROSE_PRESENT = "rough_cut_scaffold_prose_present"
ROUGH_CUT_REPEATED_AUTHOR_CREDIT = "rough_cut_repeated_author_credit"
ROUGH_CUT_HUMAN_RECEIPT_NEEDS_REVISION = "rough_cut_human_receipt_needs_revision"

BLOCKER_SEVERITIES = {"critical", "major", "blocker"}
READY_STATUS = "ready_for_operator_review"
APPROVED_STATUS = "approved_for_topic4_canary"
BLOCKED_STATUS = "blocked"

BLACK_FRAME_CODES = {"VIDEO_FLAT_FRAME_LOW_ENTROPY"}
DESYNC_CODES = {
    "PRESENTATION_AUDIO_GAP_DETECTED",
    "PRESENTATION_SLIDE_SYNC_MISMATCH",
    "PRESENTATION_VISUAL_SOURCE_MISMATCH",
    "NOTEBOOKLM_TIMING_DIVERGENCE_INCOHERENT",
    "VIDEO_FRAME_PHASH_MISMATCH",
    "VIDEO_FRAME_PIXEL_MISMATCH",
    "AUDIO_SILENCE_SPAN_EXCEEDED",
    "AUDIO_TOTAL_SILENCE_EXCEEDED",
}
ARTIFACT_TRUTH_BLACK_CODES = {"VIDEO_FLAT_FRAME_LOW_ENTROPY"}
ARTIFACT_TRUTH_SCAFFOLD_CODES = {
    "HTML_PATH_TEXT_VISUAL",
    "HTML_NO_IMAGE_OR_SVG",
    "SOURCE_GENERIC_FALLBACK",
    "SOURCE_UNRESOLVED_OR_TITLE_MISMATCHED",
    "SOURCE_MISSING_AUTHOR_OR_SPEAKER",
    "SCAFFOLD_OR_PROXY_EVIDENCE_USED",
    "LEGACY_MARKDOWN_PROOF_USED",
}
ARTIFACT_TRUTH_SCREENSHOT_CODES = {"SLIDE_FRAME_MISSING", "SLIDE_FRAME_HASH_STALE"}
ARTIFACT_TRUTH_RECEIPT_CODES = {"HUMAN_REVIEW_NOT_APPROVED"}


@dataclass(frozen=True)
class RoughCutFinding:
    code: str
    severity: str
    message: str
    path: str
    evidence: Mapping[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "path": self.path,
        }
        if self.evidence:
            payload["evidence"] = dict(self.evidence)
        return payload


@dataclass(frozen=True)
class RoughCutArtifactRef:
    role: str
    path: str
    exists: bool
    sha256: str | None
    size_bytes: int | None

    @classmethod
    def from_path(cls, role: str, path: Path) -> "RoughCutArtifactRef":
        resolved = Path(path)
        if not resolved.is_file():
            return cls(role=role, path=str(resolved), exists=False, sha256=None, size_bytes=None)
        return cls(
            role=role,
            path=str(resolved),
            exists=True,
            sha256=_sha256(resolved),
            size_bytes=resolved.stat().st_size,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "path": self.path,
            "exists": self.exists,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class RoughCutPacket:
    deck_id: str
    deck_dir: str
    generated_at: str
    artifact_refs: tuple[RoughCutArtifactRef, ...]
    brand_lock: Mapping[str, object]
    source_board: Mapping[str, object]
    storyboard_review: Mapping[str, object]
    narration_readthrough: Mapping[str, object]
    pdf_screenshots: Mapping[str, object]
    contact_sheet: Mapping[str, object]
    video_frame_samples: Mapping[str, object]
    transcript_findings: tuple[RoughCutFinding, ...]
    viewer_findings: tuple[RoughCutFinding, ...]
    human_review: Mapping[str, object]
    artifact_truth: Mapping[str, object]
    video_sync: Mapping[str, object]
    blockers: tuple[RoughCutFinding, ...]

    @property
    def status(self) -> str:
        return BLOCKED_STATUS if self.blockers else READY_STATUS

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "packet_type": PACKET_TYPE,
            "deck_id": self.deck_id,
            "deck_dir": self.deck_dir,
            "generated_at": self.generated_at,
            "status": self.status,
            "artifact_refs": [ref.to_dict() for ref in self.artifact_refs],
            "brand_lock": dict(self.brand_lock),
            "source_board": dict(self.source_board),
            "storyboard_review": dict(self.storyboard_review),
            "narration_readthrough": dict(self.narration_readthrough),
            "pdf_screenshots": dict(self.pdf_screenshots),
            "contact_sheet": dict(self.contact_sheet),
            "video_frame_samples": dict(self.video_frame_samples),
            "transcript_findings": [finding.to_dict() for finding in self.transcript_findings],
            "viewer_findings": [finding.to_dict() for finding in self.viewer_findings],
            "human_review": dict(self.human_review),
            "artifact_truth": dict(self.artifact_truth),
            "video_sync": dict(self.video_sync),
            "blockers": [finding.to_dict() for finding in self.blockers],
        }


@dataclass(frozen=True)
class RoughCutReviewResult:
    status: str
    findings: tuple[RoughCutFinding, ...]
    approved_by: str | None = None
    viewer_note: str | None = None

    @property
    def exit_code(self) -> int:
        return 0 if self.status in {READY_STATUS, APPROVED_STATUS} else 1

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "packet_type": PACKET_TYPE,
            "status": self.status,
            "findings": [finding.to_dict() for finding in self.findings],
            "blockers": [
                finding.to_dict()
                for finding in self.findings
                if finding.severity in BLOCKER_SEVERITIES
            ],
        }
        if self.approved_by:
            payload["approved_by"] = self.approved_by
        if self.viewer_note:
            payload["viewer_note"] = self.viewer_note
        return payload


def build_rough_cut_packet(
    *,
    deck_dir: Path | str,
    phase_dir: Path | str,
    brand_lock_json: Path | str,
    source_board_json: Path | str,
    storyboard_review_json: Path | str,
    narration_readthrough_md: Path | str,
    artifact_truth_json: Path | str,
    single_video_sync_json: Path | str,
    notebooklm_video_sync_json: Path | str,
) -> RoughCutPacket:
    root = Path(deck_dir).expanduser().resolve()
    phase_root = Path(phase_dir).expanduser().resolve()
    artifact_truth = _load_json(Path(artifact_truth_json), default={})
    single_sync = _load_json(Path(single_video_sync_json), default={})
    notebooklm_sync = _load_json(Path(notebooklm_video_sync_json), default={})
    video_frame_samples = _load_json(root / "phase13_canary_objective_evidence.json", default={})

    artifact_refs = _artifact_refs(root, artifact_truth_json, single_video_sync_json, notebooklm_video_sync_json)
    pdf_screenshots = _pdf_screenshot_summary(root)
    contact_sheet = _contact_sheet_summary(root)
    transcript_findings = tuple(_scan_transcripts(root))
    viewer_findings: tuple[RoughCutFinding, ...] = ()
    human_review = _human_review_summary(root / "human_review_receipt.json")

    blockers: list[RoughCutFinding] = []
    blockers.extend(_missing_artifact_blockers(artifact_refs))
    blockers.extend(_pdf_screenshot_blockers(pdf_screenshots))
    blockers.extend(_contact_sheet_blockers(contact_sheet))
    blockers.extend(transcript_findings)
    blockers.extend(_video_sync_blockers(single_sync, "single_speaker"))
    blockers.extend(_video_sync_blockers(notebooklm_sync, "notebooklm"))
    blockers.extend(_artifact_truth_blockers(artifact_truth))
    blockers.extend(_human_review_blockers(human_review))

    return RoughCutPacket(
        deck_id=root.name,
        deck_dir=str(root),
        generated_at=_isoformat_z(),
        artifact_refs=artifact_refs,
        brand_lock=_load_json(Path(brand_lock_json), default={"path": str(brand_lock_json), "status": "missing"}),
        source_board=_load_json(Path(source_board_json), default={"path": str(source_board_json), "status": "missing"}),
        storyboard_review=_load_json(
            Path(storyboard_review_json),
            default={"path": str(storyboard_review_json), "status": "missing"},
        ),
        narration_readthrough=_narration_readthrough_summary(Path(narration_readthrough_md), phase_root),
        pdf_screenshots=pdf_screenshots,
        contact_sheet=contact_sheet,
        video_frame_samples=video_frame_samples,
        transcript_findings=transcript_findings,
        viewer_findings=viewer_findings,
        human_review=human_review,
        artifact_truth=artifact_truth,
        video_sync={"single_speaker": single_sync, "notebooklm": notebooklm_sync},
        blockers=tuple(_dedupe_findings(blockers)),
    )


def review_rough_cut_packet(
    packet: Mapping[str, Any],
    *,
    approved_by: str | None = None,
    viewer_note: str | None = None,
) -> RoughCutReviewResult:
    findings = [RoughCutFinding(**finding) for finding in _coerce_finding_dicts(packet.get("blockers"))]
    findings.extend(
        RoughCutFinding(**finding)
        for finding in _coerce_finding_dicts(packet.get("viewer_findings"))
        if str(finding.get("severity", "")).lower() in BLOCKER_SEVERITIES
    )
    human_review = packet.get("human_review")
    if isinstance(human_review, Mapping):
        findings.extend(_human_review_blockers(human_review))
    status = BLOCKED_STATUS if findings else (APPROVED_STATUS if approved_by else READY_STATUS)
    return RoughCutReviewResult(
        status=status,
        findings=tuple(_dedupe_findings(findings)),
        approved_by=approved_by,
        viewer_note=viewer_note,
    )


def packet_to_markdown(packet: Mapping[str, Any]) -> str:
    blockers = _coerce_finding_dicts(packet.get("blockers"))
    refs = packet.get("artifact_refs") if isinstance(packet.get("artifact_refs"), list) else []
    lines = [
        "# Rough-Cut Video Review",
        "",
        f"Deck: `{packet.get('deck_id', '')}`",
        f"Status: `{packet.get('status', 'unknown')}`",
        f"Blockers: `{len(blockers)}`",
        "",
        "## Artifact Refs",
        "",
    ]
    for ref in refs:
        if isinstance(ref, Mapping):
            lines.append(
                f"- `{ref.get('role')}` `{ref.get('path')}` exists={ref.get('exists')} sha256={ref.get('sha256')}"
            )
    lines.extend(["", "## Human Review", ""])
    human = packet.get("human_review") if isinstance(packet.get("human_review"), Mapping) else {}
    lines.append(f"- receipt_gate: `{human.get('receipt_gate', 'unknown')}`")
    lines.append(f"- state: `{human.get('state', 'unknown')}`")
    if blockers:
        lines.extend(["", "## Blockers", ""])
        for blocker in blockers:
            lines.append(f"- `{blocker.get('code')}` {blocker.get('message')} ({blocker.get('path')})")
    return "\n".join(lines) + "\n"


def review_to_markdown(result: Mapping[str, Any]) -> str:
    blockers = result.get("blockers") if isinstance(result.get("blockers"), list) else []
    lines = [
        "# Rough-Cut Review Result",
        "",
        f"Status: `{result.get('status', 'unknown')}`",
        f"Blockers: `{len(blockers)}`",
    ]
    if result.get("approved_by"):
        lines.append(f"Approved by: `{result.get('approved_by')}`")
    if result.get("viewer_note"):
        lines.extend(["", "## Viewer Note", "", str(result.get("viewer_note"))])
    if blockers:
        lines.extend(["", "## Blockers", ""])
        for blocker in blockers:
            if isinstance(blocker, Mapping):
                lines.append(f"- `{blocker.get('code')}` {blocker.get('message')}")
    return "\n".join(lines) + "\n"


def _artifact_refs(
    root: Path,
    artifact_truth_json: Path | str,
    single_video_sync_json: Path | str,
    notebooklm_video_sync_json: Path | str,
) -> tuple[RoughCutArtifactRef, ...]:
    refs = [
        ("presentation_html", root / "presentation.html"),
        ("presentation_pdf", root / "presentation.pdf"),
        ("render_contact_sheet", root / "render_contact_sheet.png"),
        ("video_contact_sheet", root / "phase13_canary_frames" / "contact_sheet.png"),
        ("single_presenter_video", root / "presentation_video.mp4"),
        ("notebooklm_video", root / "presentation_video_notebooklm.mp4"),
        ("single_presenter_script", root / "6_presenter_script.md"),
        ("notebooklm_script", root / "6_presenter_script_notebooklm_2host.md"),
        ("artifact_truth_report", Path(artifact_truth_json)),
        ("single_video_sync_report", Path(single_video_sync_json)),
        ("notebooklm_video_sync_report", Path(notebooklm_video_sync_json)),
    ]
    return tuple(RoughCutArtifactRef.from_path(role, path) for role, path in refs)


def _missing_artifact_blockers(refs: Sequence[RoughCutArtifactRef]) -> list[RoughCutFinding]:
    blockers: list[RoughCutFinding] = []
    for ref in refs:
        if ref.exists:
            continue
        if ref.role in {"single_presenter_video", "notebooklm_video"}:
            code = ROUGH_CUT_MISSING_MP4
        elif ref.role in {"render_contact_sheet", "video_contact_sheet"}:
            code = ROUGH_CUT_MISSING_CONTACT_SHEET
        elif ref.role == "presentation_pdf":
            code = ROUGH_CUT_MISSING_PDF_SCREENSHOTS
        else:
            code = ROUGH_CUT_SCAFFOLD_PROSE_PRESENT
        blockers.append(
            RoughCutFinding(
                code=code,
                severity="blocker",
                message=f"required rough-cut artifact is missing: {ref.role}",
                path=ref.path,
                evidence={"role": ref.role},
            )
        )
    return blockers


def _pdf_screenshot_summary(root: Path) -> dict[str, object]:
    frame_dir = root / "rendered_frames"
    frames = sorted(frame_dir.glob("slide_*.png")) if frame_dir.is_dir() else []
    render_spec = _load_json(root / "render_spec.json", default={})
    expected = len(render_spec.get("slides", [])) if isinstance(render_spec.get("slides"), list) else 0
    return {
        "frame_dir": str(frame_dir),
        "expected_count": expected,
        "actual_count": len(frames),
        "paths": [str(frame) for frame in frames],
        "status": "pass" if frames and (not expected or len(frames) >= expected) else "missing",
    }


def _pdf_screenshot_blockers(summary: Mapping[str, object]) -> list[RoughCutFinding]:
    expected = int(summary.get("expected_count") or 0)
    actual = int(summary.get("actual_count") or 0)
    if actual and (not expected or actual >= expected):
        return []
    return [
        RoughCutFinding(
            code=ROUGH_CUT_MISSING_PDF_SCREENSHOTS,
            severity="blocker",
            message="rendered PDF/screenshot frame set is missing or incomplete",
            path=str(summary.get("frame_dir") or "rendered_frames"),
            evidence={"expected_count": expected, "actual_count": actual},
        )
    ]


def _contact_sheet_summary(root: Path) -> dict[str, object]:
    render = root / "render_contact_sheet.png"
    video = root / "phase13_canary_frames" / "contact_sheet.png"
    return {
        "render_contact_sheet": str(render),
        "render_contact_sheet_exists": render.is_file(),
        "video_contact_sheet": str(video),
        "video_contact_sheet_exists": video.is_file(),
        "status": "pass" if render.is_file() and video.is_file() else "missing",
    }


def _contact_sheet_blockers(summary: Mapping[str, object]) -> list[RoughCutFinding]:
    blockers: list[RoughCutFinding] = []
    for key in ("render_contact_sheet", "video_contact_sheet"):
        if summary.get(f"{key}_exists"):
            continue
        blockers.append(
            RoughCutFinding(
                code=ROUGH_CUT_MISSING_CONTACT_SHEET,
                severity="blocker",
                message=f"{key} is missing",
                path=str(summary.get(key) or key),
            )
        )
    return blockers


def _scan_transcripts(root: Path) -> list[RoughCutFinding]:
    findings: list[RoughCutFinding] = []
    script_paths = [
        root / "6_presenter_script.md",
        root / "6_presenter_script_notebooklm_2host.md",
    ]
    author_credit_count = 0
    for path in script_paths:
        if not path.is_file():
            findings.append(
                RoughCutFinding(
                    code=ROUGH_CUT_SCAFFOLD_PROSE_PRESENT,
                    severity="blocker",
                    message="required transcript is missing",
                    path=str(path),
                )
            )
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        author_credit_count += len(re.findall(r"Prepared by Nikolaos Vasiloglou", text))
        if re.search(r"\bHost A frames\b|\bHost B connects\b", text):
            findings.append(
                RoughCutFinding(
                    code=ROUGH_CUT_MECHANICAL_DIALOGUE,
                    severity="blocker",
                    message="mechanical NotebookLM host dialogue remains in transcript",
                    path=str(path),
                    evidence={"pattern": "Host A frames|Host B connects"},
                )
            )
        if re.search(r"Use the visual role|Structured visual asset|visual-ref", text):
            findings.append(
                RoughCutFinding(
                    code=ROUGH_CUT_SCAFFOLD_PROSE_PRESENT,
                    severity="blocker",
                    message="scaffold prose remains in transcript",
                    path=str(path),
                )
            )
        if re.search(r"\bthe diagram shows\b", text, flags=re.IGNORECASE):
            findings.append(
                RoughCutFinding(
                    code=ROUGH_CUT_VISUAL_CLAIM_MISMATCH,
                    severity="blocker",
                    message="transcript references a diagram claim without proof-object grounding",
                    path=str(path),
                    evidence={"pattern": "the diagram shows"},
                )
            )
    if author_credit_count > 1:
        findings.append(
            RoughCutFinding(
                code=ROUGH_CUT_REPEATED_AUTHOR_CREDIT,
                severity="blocker",
                message="repeated prepared-by credit remains in content narration",
                path=str(root),
                evidence={"count": author_credit_count},
            )
        )
    elif author_credit_count == 1:
        findings.append(
            RoughCutFinding(
                code=ROUGH_CUT_SCAFFOLD_PROSE_PRESENT,
                severity="blocker",
                message="prepared-by scaffold prose remains in transcript",
                path=str(root),
                evidence={"count": author_credit_count},
            )
        )
    return findings


def _video_sync_blockers(payload: Mapping[str, object], style: str) -> list[RoughCutFinding]:
    blockers: list[RoughCutFinding] = []
    for error in _coerce_finding_dicts(payload.get("errors")):
        code = str(error.get("code") or "")
        mapped = None
        if code in BLACK_FRAME_CODES or "black" in str(error.get("message", "")).lower():
            mapped = ROUGH_CUT_VIDEO_BLACK_FRAME
        elif code in DESYNC_CODES:
            mapped = ROUGH_CUT_VIDEO_DESYNC
        if mapped:
            blockers.append(
                RoughCutFinding(
                    code=mapped,
                    severity="blocker",
                    message=f"{style} video sync report contains {code}",
                    path=str(error.get("path") or style),
                    evidence={"source_code": code, "style": style},
                )
            )
    return blockers


def _artifact_truth_blockers(payload: Mapping[str, object]) -> list[RoughCutFinding]:
    blockers: list[RoughCutFinding] = []
    for finding in _coerce_finding_dicts(payload.get("blockers")):
        code = str(finding.get("code") or "")
        if code in ARTIFACT_TRUTH_RECEIPT_CODES:
            continue
        if code in ARTIFACT_TRUTH_SCREENSHOT_CODES:
            mapped = ROUGH_CUT_MISSING_PDF_SCREENSHOTS
        elif code in ARTIFACT_TRUTH_BLACK_CODES:
            mapped = ROUGH_CUT_VIDEO_BLACK_FRAME
        elif code in DESYNC_CODES:
            mapped = ROUGH_CUT_VIDEO_DESYNC
        elif code in ARTIFACT_TRUTH_SCAFFOLD_CODES:
            mapped = ROUGH_CUT_SCAFFOLD_PROSE_PRESENT
        else:
            mapped = ROUGH_CUT_SCAFFOLD_PROSE_PRESENT
        blockers.append(
            RoughCutFinding(
                code=mapped,
                severity="blocker",
                message=f"artifact truth report contains {code}",
                path=str(finding.get("path") or "artifact_truth_report.json"),
                evidence={"source_code": code},
            )
        )
    return blockers


def _human_review_summary(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {
            "path": str(path),
            "exists": False,
            "state": "missing",
            "receipt_gate": "pending_human_review",
            "blocks_rollout": True,
            "blocks_rough_cut": False,
        }
    payload = _load_json(path, default={})
    status = str(payload.get("status") or payload.get("decision") or "").strip() or "unknown"
    needs_revision = "needs_revision" in status
    approved = status == APPROVED_STATUS
    return {
        "path": str(path),
        "exists": True,
        "state": status,
        "receipt_gate": "approved" if approved else "pending_human_review",
        "blocks_rollout": not approved,
        "blocks_rough_cut": bool(payload.get("blocks_rough_cut")) or needs_revision and bool(payload.get("used_as_rough_cut_approval")),
        "sha256": _sha256(path),
    }


def _human_review_blockers(payload: Mapping[str, object]) -> list[RoughCutFinding]:
    state = str(payload.get("state") or "").lower()
    if "needs_revision" not in state or not payload.get("blocks_rough_cut"):
        return []
    return [
        RoughCutFinding(
            code=ROUGH_CUT_HUMAN_RECEIPT_NEEDS_REVISION,
            severity="blocker",
            message="needs_revision human receipt cannot approve the rough cut",
            path=str(payload.get("path") or "human_review_receipt.json"),
            evidence={"state": state},
        )
    ]


def _narration_readthrough_summary(path: Path, phase_root: Path) -> dict[str, object]:
    candidate = path if path.is_file() else phase_root / path.name
    if not candidate.is_file():
        return {"path": str(path), "exists": False, "sha256": None}
    return {"path": str(candidate), "exists": True, "sha256": _sha256(candidate)}


def _coerce_finding_dicts(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _dedupe_findings(findings: Iterable[RoughCutFinding]) -> list[RoughCutFinding]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[RoughCutFinding] = []
    for finding in findings:
        key = (finding.code, finding.path, finding.message)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(finding)
    return deduped


def _load_json(path: Path, *, default: Mapping[str, object]) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(default)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _isoformat_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

