from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import timezone, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
BRAND_ASSET_DIR = PROJECT_ROOT / "assets" / "brand" / "relationalai"
BRAND_ASSETS = {
    "bug": BRAND_ASSET_DIR / "bug_orange.png",
    "decorative_art": BRAND_ASSET_DIR / "title_decorative_art.png",
    "logo": BRAND_ASSET_DIR / "logo_white.png",
}
BRAND_REFERENCE_NAME = "relationalai-neurips2025-reference"

CSS_VARIABLES: dict[str, str] = {
    "--navy": "#1a1a2e",
    "--rai-blue": "#058DC7",
    "--ice-blue": "#a8d8ea",
    "--rai-teal": "#158158",
    "--rai-orange": "#ED561B",
    "--rai-green": "#50B432",
}

RENDERER_GEOMETRY: dict[str, dict[str, Any]] = {
    "title_logo": {
        "width": 309,
        "height": 48,
        "top_css_px": 26,
        "left_css_px": 77,
        "background_size": "contain",
    },
    "decorative_art": {
        "width": 520,
        "height": 530,
        "top_css_px": -30,
        "right_css_px": -30,
        "opacity": 0.8,
    },
    "content_brand_bug": {
        "width": 54,
        "height": 40,
        "right_css_px": 24,
        "bottom_css_px": 18,
        "background_size": "contain",
    },
}

FAKE_INLINE_TITLE_LOGO_SVG = "fake_inline_title_logo_svg"
TEXT_ONLY_RAI_BRAND_BUG = "text_only_rai_brand_bug"
TEMPLATE_METADATA_MISSING = "template_metadata_missing"
TEMPLATE_SHA256_MISSING = "template_sha256_missing"
BRAND_ASSET_HASH_MISSING = "brand_asset_hash_missing"
SCAFFOLD_PROSE_PRESENT = "scaffold_prose_present"
REPEATED_AUTHOR_CREDIT_ON_CONTENT_SLIDE = "repeated_author_credit_on_content_slide"


@dataclass(frozen=True)
class BrandTemplateFinding:
    code: str
    severity: str
    message: str
    path: str


@dataclass(frozen=True)
class BrandAssetRef:
    role: str
    path: str
    sha256: str


@dataclass(frozen=True)
class BrandTemplateLock:
    schema_version: int
    lock_type: str
    status: str
    reference_name: str
    reference_deck_path: str
    reference_deck_sha256: str
    template_name: str
    template_sha256: str
    asset_refs: tuple[BrandAssetRef, ...]
    renderer_geometry: dict[str, Any]
    css_variables: dict[str, str]
    findings: tuple[BrandTemplateFinding, ...]
    generated_at: str


def brand_assets_sha256() -> str:
    h = hashlib.sha256()
    for key in sorted(BRAND_ASSETS):
        path = BRAND_ASSETS[key]
        if not path.exists():
            return ""
        h.update(key.encode("utf-8"))
        h.update(path.read_bytes())
    return h.hexdigest()


def _has_inline_svg_title_logo(html: str) -> bool:
    idx = html.find('class="title-logo"')
    if idx == -1:
        return False
    snippet = html[idx : idx + 2000]
    svg_pos = snippet.find("<svg")
    img_pos = snippet.find("<img")
    if svg_pos == -1:
        return False
    return img_pos == -1 or svg_pos < img_pos


def _has_text_only_rai_bug(html: str) -> bool:
    for class_name in ('class="content-bug"', 'class="content-brand-bug"'):
        idx = html.find(class_name)
        if idx == -1:
            continue
        close = html.find("</div>", idx)
        if close == -1:
            close = idx + 500
        snippet = html[idx:close]
        if "<img" in snippet:
            continue
        if "RAI" in snippet:
            return True
    return False


def _extract_non_title_body(html: str) -> str:
    lower = html.lower()
    title_markers = ["title-slide", 'class="title"', 'slide-type: title', 'slide_type: title']
    body_start = lower.find("<body")
    if body_start == -1:
        return html
    sections: list[str] = []
    section_tag = "<section"
    pos = html.find(section_tag, body_start)
    while pos != -1:
        end = html.find("</section>", pos)
        if end == -1:
            end = len(html)
        chunk = html[pos : end + len("</section>")]
        chunk_lower = chunk.lower()
        is_title = any(marker in chunk_lower for marker in title_markers)
        if not is_title:
            sections.append(chunk)
        pos = html.find(section_tag, end)
    return "\n".join(sections) if sections else html[body_start:]


def validate_brand_template_lock(
    html_content: str,
    assets_sha256: str,
    reference_name: str,
    reference_deck_path: str = "",
    reference_deck_sha256: str = "",
) -> BrandTemplateLock:
    findings: list[BrandTemplateFinding] = []

    if 'name="presentation-template"' not in html_content:
        findings.append(BrandTemplateFinding(
            code=TEMPLATE_METADATA_MISSING,
            severity="blocker",
            message="Missing <meta name=\"presentation-template\"> tag.",
            path="",
        ))

    if 'name="presentation-template-sha256"' not in html_content:
        findings.append(BrandTemplateFinding(
            code=TEMPLATE_SHA256_MISSING,
            severity="blocker",
            message="Missing <meta name=\"presentation-template-sha256\"> tag.",
            path="",
        ))

    if not assets_sha256:
        findings.append(BrandTemplateFinding(
            code=BRAND_ASSET_HASH_MISSING,
            severity="blocker",
            message="Brand asset SHA-256 is empty — canonical assets missing or not hashed.",
            path="",
        ))
    elif assets_sha256 not in html_content:
        findings.append(BrandTemplateFinding(
            code=BRAND_ASSET_HASH_MISSING,
            severity="blocker",
            message="Brand asset SHA-256 not embedded in deck HTML — deck was not built from canonical brand assets.",
            path="",
        ))

    if _has_inline_svg_title_logo(html_content):
        findings.append(BrandTemplateFinding(
            code=FAKE_INLINE_TITLE_LOGO_SVG,
            severity="blocker",
            message="title-logo div contains inline <svg> instead of image-backed <img> tag.",
            path="",
        ))

    if _has_text_only_rai_bug(html_content):
        findings.append(BrandTemplateFinding(
            code=TEXT_ONLY_RAI_BRAND_BUG,
            severity="blocker",
            message="content-bug div contains literal 'RAI' text without an <img> tag.",
            path="",
        ))

    if "Use the visual role" in html_content:
        findings.append(BrandTemplateFinding(
            code=SCAFFOLD_PROSE_PRESENT,
            severity="blocker",
            message="Scaffold prose 'Use the visual role' found in HTML content.",
            path="",
        ))

    non_title_body = _extract_non_title_body(html_content)
    if "Prepared by Nikolaos Vasiloglou" in non_title_body:
        findings.append(BrandTemplateFinding(
            code=REPEATED_AUTHOR_CREDIT_ON_CONTENT_SLIDE,
            severity="blocker",
            message="'Prepared by Nikolaos Vasiloglou' found on a non-title slide.",
            path="",
        ))

    status = "pass" if all(f.severity != "blocker" for f in findings) else "fail"

    asset_refs = tuple(
        BrandAssetRef(role=role, path=str(path), sha256="")
        for role, path in BRAND_ASSETS.items()
    )

    return BrandTemplateLock(
        schema_version=1,
        lock_type="conference_pipeline.presentation.brand_template_lock",
        status=status,
        reference_name=reference_name,
        reference_deck_path=reference_deck_path,
        reference_deck_sha256=reference_deck_sha256,
        template_name=reference_name,
        template_sha256=assets_sha256,
        asset_refs=asset_refs,
        renderer_geometry=dict(RENDERER_GEOMETRY),
        css_variables=dict(CSS_VARIABLES),
        findings=tuple(findings),
        generated_at=datetime.now(tz=timezone.utc).isoformat(),
    )
