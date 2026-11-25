#!/usr/bin/env python3
"""Author figure specs for a deck: what each figure claims, never how it looks.

The model's whole job here is the declarative spec -- archetype, entities
verbatim, numbers bound to numeric-claims ledger rows, emphasis, a takeaway.
Layout, colour, typography and marker geometry belong to the compiler
(``conference_pipeline.figures``), so the prompt forbids them and the validator
rejects them.

    python scripts/presentations/generate_figure_specs.py \
        --deck-dir conferences/<c>/decks/<id> --workspace conferences/<c> [--dry-run]

``--dry-run`` writes the prompts and a deterministic fallback spec per slide
without calling a provider, so the wiring can be proven for free.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextvars
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from conference_pipeline.presentation.claim_fidelity import source_record  # noqa: E402
from conference_pipeline.figures.errors import FigureSpecError  # noqa: E402
from conference_pipeline.figures.examples import EXAMPLES  # noqa: E402
from conference_pipeline.figures.grounding import filter_components, filter_edges  # noqa: E402
from conference_pipeline.figures.ledger import NumericLedger, build_ledger, load_ledger  # noqa: E402
from conference_pipeline.figures.lint import rule_summaries, title_repeats_slide_title, unsupported_outcomes, source_fidelity_errors  # noqa: E402
from conference_pipeline.figures.compiler import compile_figure  # noqa: E402
from conference_pipeline.figures.selection import parse_visual_role, select_archetype  # noqa: E402
from conference_pipeline.figures.glyphs import GLYPH_MODIFIERS  # noqa: E402
from conference_pipeline.figures.spec import (  # noqa: E402
    ARCHETYPES,
    CONTENT_KEYS,
    CURVE_SHAPES,
    FAMILY_SCHEMATIC,
    FAMILY_VISUAL,
    GLYPHS,
    QUOTA_EXEMPT_FAMILIES,
    SPEC_VERSION,
    archetype_catalog,
    content_labels,
    load_spec,
    validate_spec,
)
from conference_pipeline.model_governance.llm_client import (  # noqa: E402
    complete,
    credential_names,
    has_credentials,
    resolve_selection,
)
from conference_pipeline.provider_receipts import provider_receipt_context  # noqa: E402

from conference_pipeline.figures.selection import PICTURE_ARCHETYPES, relation_heavy  # noqa: E402
from conference_pipeline.figures.errors import FigureCompilerError  # noqa: E402

TASK_ID = "presentation.figure_spec_authoring"
SPEC_DIR_NAME = "figure_specs"
LEDGER_NAME = "numeric_claims_ledger.json"
RECEIPT_NAME = "figure_spec_receipt.json"
# Digests of specs that passed authoring (validation, lints, compile) or that the
# operator accepted. Kept apart from the receipt, which a blocked run overwrites.
ACCEPTED_DIGESTS_NAME = "figure_spec_accepted_digests.json"


def spec_digest(payload: dict[str, Any]) -> str:
    """The same canonical digest patch_figure_spec.py records as before/after_sha256."""

    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def spec_standing(spec_dir: Path, figure_id: str, digest: str, accepted: dict[str, str]) -> str:
    """How the spec on disk came to be: "accepted", "operator_patched" or "".

    accepted: unchanged since it passed authoring, or the operator's latest
    decision on it was "accept". operator_patched: its content is the result of
    the operator's latest patch/replace.
    """

    entries: list[dict[str, Any]] = []
    patch_log = spec_dir / f"{figure_id}.patch.json"
    if patch_log.is_file():
        try:
            loaded = json.loads(patch_log.read_text(encoding="utf-8"))
            entries = loaded if isinstance(loaded, list) else []
        except (OSError, json.JSONDecodeError):
            entries = []
    last = entries[-1] if entries and isinstance(entries[-1], dict) else {}
    if last.get("outcome") == "accept" and digest in (last.get("before_sha256"), last.get("after_sha256")):
        return "accepted"
    if last.get("outcome") in ("patch", "replace") and last.get("after_sha256") == digest:
        return "operator_patched"
    if accepted.get(figure_id) == digest:
        return "accepted"
    return ""
MAX_REPAIR_ATTEMPTS = 2  # validation errors are mechanical and repair is cheap


def _write_spec_preserving_refusal(spec_path: Path, payload: dict[str, Any]) -> None:
    """Keep a refused predecessor when a nondeterministic retry replaces it."""

    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if spec_path.is_file():
        previous = spec_path.read_text(encoding="utf-8")
        if previous != encoded:
            refused = False
            try:
                compile_figure(json.loads(previous))
            except (FigureCompilerError, json.JSONDecodeError):
                refused = True
            if refused:
                digest = hashlib.sha256(previous.encode("utf-8")).hexdigest()[:12]
                history = spec_path.parent / "refused" / f"{spec_path.stem}.{digest}.json"
                history.parent.mkdir(parents=True, exist_ok=True)
                if not history.exists():
                    history.write_text(previous, encoding="utf-8")
    spec_path.write_text(encoded, encoding="utf-8")
# Stamped into every --dry-run spec. A placeholder is structurally valid, so
# nothing downstream would notice it -- which is exactly how boilerplate ships.
PLACEHOLDER_NOTE = "placeholder spec (dry run)"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _slide_rows(deck_contract: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for slide in deck_contract.get("slides") or []:
        if not isinstance(slide, dict):
            continue
        if str(slide.get("slide_kind") or "").strip().lower() in {"events", "resources"}:
            continue
        rows.append(slide)
    return rows


def _required_entities(slide: dict[str, Any]) -> list[str]:
    entities: list[str] = []
    for key in ("required_entities", "required_labels", "visible_entities"):
        value = slide.get(key)
        if isinstance(value, list):
            entities.extend(str(item).strip() for item in value if str(item).strip())
    return list(dict.fromkeys(entities))


def build_prompt(
    *,
    slide: dict[str, Any],
    archetype: str,
    ledger: NumericLedger,
    ledger_hits: list[Any],
) -> str:
    """The spec-author prompt for one slide."""

    definition = ARCHETYPES[archetype]
    example = json.dumps(EXAMPLES.get(archetype, {}), indent=2)
    catalog_row = next(row for row in archetype_catalog() if row["archetype"] == archetype)
    rows_text = (
        "\n".join(
            f"- {claim.claim_id}: {claim.value}{claim.unit} ({claim.metric or 'unlabelled metric'}) "
            f"-- \"{claim.span[:180]}\" [source {claim.source_id}]"
            for claim in ledger_hits
        )
        or "- (no numeric-claims ledger row matches this slide)"
    )
    entities = _required_entities(slide)
    # Closed vocabularies the validator enforces. Without them in the prompt the
    # author has to guess a glyph or shape name, and a guessed name is a
    # rejected spec and a wasted call.
    vocabulary = ""
    if definition.family == FAMILY_VISUAL:
        vocabulary = (
            "\n  give every panel a `depicts`: a short noun phrase naming what the "
            "picture stands for (\"compressed video frame\", \"jailbreak prompt\", "
            "\"attention window\"). The compiler picks the drawing from it."
            f"\n  glyph names, if you want to override the choice: {GLYPHS}"
            f"\n  modifiers you may name: {GLYPH_MODIFIERS}"
            "\n  a panel with no depicts and no glyph draws a photo: a generic "
            "landscape thumbnail, the same one on every panel."
        )
        if archetype == "modality_fusion":
            vocabulary += (
                "\n  competing mechanisms: fusion.alternatives = [{id, label, output: {label}}, ...] "
                "(two or three); the same modalities feed each, and each draws its own result."
                "\n  several results of ONE mechanism: fusion.output may be a list of up to three {label}."
            )
    elif definition.family == FAMILY_SCHEMATIC:
        vocabulary = f"\n  curve shapes you may use (exactly these): {CURVE_SHAPES}"
    if archetype == "hub_spoke":
        # Without this the author cannot say which way a spoke points, so a hub
        # that is the OUTCOME of its spokes was drawn radiating into its own
        # causes -- well-formed arrows saying the opposite of the takeaway.
        vocabulary += (
            "\n  each spoke may set `direction`: \"to_hub\" when the spoke FEEDS the "
            "hub (causes, inputs, levers that converge on it), \"from_hub\" (the "
            "default) when the hub PRODUCES the spoke (consequences, outputs)."
            "\n  choose per spoke, and make it agree with the takeaway: if the claim "
            "is that several things converge on one outcome, those spokes are "
            "\"to_hub\" and the hub is the outcome."
        )
    if archetype == "feature_matrix":
        vocabulary += (
            "\n  a cell is \"yes\", \"no\" or \"partial\" ONLY when an assigned source states that "
            "property for that row; a cell no source states is \"not_reported\" (drawn as a dash). "
            "Never fill a cell by inference, and never add a row or column the sources do not "
            "describe: a matrix with one real row is not a comparison."
        )
    if archetype == "quadrant_map":
        vocabulary += (
            "\n  a point's position is a measurement: grounding.kind is 'data' and grounding.ledger_rows "
            "lists one numeric-claims ledger row per point, with x/y set from it. A point with no "
            "row is not drawn; a positioning claim with no measured rows is 'landscape_regions'."
        )
    if archetype == "flow":
        vocabulary += (
            "\n  use edge `kind: \"feedback\"` for an explicit return from a later stage "
            "to an earlier one; it gets a dashed outer lane and an arrowhead. "
            "Include the forward path too so the loop closes."
            "\n  use edge `kind: \"parallel\"` for complementary or simultaneous "
            "methods, not a causal step; it draws a headless dashed relation "
            "and does not force one method into the next layer. Label the "
            "relation so the reader knows why the methods are paired."
            "\n  when two or more branches combine, include an explicit node with "
            "`kind: \"merge\"` and a short label naming what combines; connect "
            "each branch to that node. The merge may itself be the final result, "
            "or may connect to a separate result. Do not "
            "draw two branches that appear to meet at an unlabelled point."
        )

    # The blueprint's components and edges ARE the figure's content. They were
    # being handed over buried in one 900-character string, which is how a spec
    # ends up with stages the blueprint never named and none of the edges it did.
    role = parse_visual_role(str(slide.get("visual_role") or ""))
    # The brief was written with the whole evidence digest in view and a quota
    # of components to fill; what this slide's sources do not name is dropped
    # here and listed as off-limits, so the author is never told to draw it.
    kept_components, dropped_components = filter_components(
        role["components"], slide, depiction=role["prefix"]
    )
    kept_edges = filter_edges([f"{a} -> {b}" for a, b in role["edges"]], kept_components)
    components = "\n".join(f"    - {item}" for item in kept_components) or "    (none given)"
    edges = "\n".join(f"    - {item}" for item in kept_edges) or "    (none given)"
    dropped = (
        "\n  NOT supported by this slide's sources -- do not draw these or anything like them:\n"
        + "\n".join(f"    - {item}" for item in dropped_components)
        if dropped_components else ""
    )
    claims = "\n".join(f"    - {claim}" for claim in (slide.get("claims") or [])[:4]) or "    (none)"
    bullets = "\n".join(
        f"    - {line}" for line in (slide.get("visible_text") or [])[:6]
    ) or "    (none)"
    rules = "\n".join(f"- {line}" for line in rule_summaries())

    return f"""You are authoring a FIGURE SPEC: a JSON object describing what one slide's
figure claims. A deterministic compiler turns it into an SVG.

You decide WHAT the figure says. You do NOT decide how it looks. Never emit
coordinates, sizes, colours, fonts, marker geometry, or a renderer choice --
the compiler owns all of that and the validator rejects specs that carry it.

SLIDE
  slide_id: {slide.get('slide_id')}
  section: {slide.get('section_id') or ''}
  title: {slide.get('title') or ''}
  what this slide argues:
{claims}
  bullets the reader will see (reuse their wording for entity names):
{bullets}
  narration beats: {json.dumps(slide.get('narration_hooks') or [])[:600]}

THE SLIDE'S SOURCES AND WHAT EACH SUPPORTS (the approved evidence)
{approved_source_lines(slide)}
Every reported outcome/result node must be explicitly stated by its source contribution_note. Position papers are
"argues/proposes", never demonstrated results. Join different sources' methods/components (edges, nesting or sequential steps)
ONLY with the optional top-level synthesis_label, a short caption such as
"Talk's synthesis" or "Illustration, not a reported experiment". Otherwise draw
separate alternatives/branches. A labelled combination is the talk's synthesis.
For feature_matrix/evidence tables, use "no" only when a bound note states the
negative; missing detail must be "not_reported". Never assert paper-wide absence
in claims/takeaways ("No control is tested at deployment scale yet", "none has
been shown"). A slide claim or supports sentence is not source evidence.
Name only systems, methods, benchmarks, works, people and results that appear in
the slide text or the source lines above. Never introduce another name, row,
point or comparison, and never place items at coordinates nobody measured.

THE FIGURE THE BLUEPRINT ASKED FOR
  intent: {role['prefix']} -- {role['gloss']}
  components:
{components}
  relations between them:
{edges}{dropped}
  entities that MUST appear verbatim: {json.dumps(entities)}

Draw the supported components and relations above; they are candidates from the
blueprint, and the sources rule. Carry fewer items rather than add one the sources
do not name. If a relation the blueprint names cannot be shown by this archetype,
say so rather than drop it.

ARCHETYPE TO USE: {archetype} -- {definition.summary}
  required content keys: {catalog_row['required_content']}
  content keys this archetype reads (anything else is never drawn): {sorted(CONTENT_KEYS[archetype])}
  budgets: {definition.budget_line()}
  (measured on this canvas: every pair above compiles, and one more item or one
   longer label does not. Fewer items buy longer labels, so choose both together.)
  allowed grounding kinds: {catalog_row['grounding_kinds']}{vocabulary}

BUDGETS ARE HARD LIMITS, not guidance. max_label_chars counts characters in a
single label; if a label is longer, SHORTEN THE WORDS -- do not assume the
figure will shrink to fit, because it will not: the spec is rejected.

STRUCTURE RULES (the spec is refused if it breaks one)
{rules}

NUMERIC-CLAIMS LEDGER ROWS AVAILABLE TO THIS SLIDE
{rows_text}

GROUNDING RULES
- Plot a number ONLY if a ledger row above states it, and list that row id in
  grounding.ledger_rows. Numbers you cannot cite do not go in the figure.
- With no usable rows, use grounding.kind "schematic" and an archetype that
  makes no numeric claim. Never invent illustrative values or axis ticks.
- A schematic figure names quantities on its axes, never numbers.

WRITE
A single JSON object, spec_version "{SPEC_VERSION}", with: figure_id, slide_id,
archetype, claim (one sentence), takeaway (at most 12 words), title,
grounding, emphasis, required_entities, content. No prose, no code fences.
emphasis is an object: {{"highlight": [ids of the winners the takeaway names]}}. A bare list or a highlight_ids key is read too, but say it this way.

WORKED EXAMPLE OF THIS ARCHETYPE
{example}
"""


CRITIC_TASK_ID = "presentation.figure_spec_critic"
CRITIC_RUBRIC_VERSION = "figure-spec-critic/3-adversarial"
CRITIC_EST_USD_PER_CALL = 0.04
# The adversarial critic is always on and reads each spec at most this many
# times: critique, re-author on a major issue, critique again. A figure still
# carrying a major issue after the last round is reported for the operator,
# never looped on (operator rule: three review rounds, then a person decides).
MAX_CRITIC_ROUNDS = 3
CRITIC_SEVERITIES = ("major", "minor")
CRITIC_CODES = (
    "unsupported_number",
    "unsupported_label",
    "misattribution",
    "overclaim",
    "sequence_implies_causality",
    "alternatives_as_sequence",
    "orphan_element",
    "wrong_archetype",
    "incommensurable_axes",
    "unlabelled_branch",
    "emphasis_off_claim",
    "composed_sources",
    "position_as_result",
    "depiction_mismatch",
    "claim_not_encoded",
    "other",
)


def _merge_source_fields(*rows: dict[str, Any]) -> dict[str, Any]:
    """Merge same-work metadata without erasing evidence with empty fields."""
    return {key: value for row in rows for key, value in row.items()
            if value is not None and value != "" and value != []}


def _source_key(row: dict[str, Any]) -> str:
    return str(row.get("label") or row.get("source_id") or "")


def approved_source_lines(slide: dict[str, Any]) -> str:
    """The slide's sources with the blueprint's per-slide 'supports' sentence.

    Attached to the slide as ``approved_sources`` from the approved blueprint
    (see ``attach_approved_sources``); falls back to the deck contract's
    mentions, which carry titles and people only.
    """

    lines: list[str] = []
    approved = [a for a in slide.get("approved_sources") or [] if isinstance(a, dict)]
    plans = {_source_key(row): row for row in slide.get("source_plan") or [] if isinstance(row, dict)}
    rows = approved or [
        m for m in (slide.get("named_paper_talk_speaker_author_mentions")
                    or slide.get("speaker_author_work_mentions") or []) if isinstance(m, dict)
    ] or list(plans.values())
    for row in rows:
        row = _merge_source_fields(plans.get(_source_key(row), {}), row)
        record = source_record(row)
        lines.append("SOURCE BLOCK\n" + json.dumps({**record, "supports": row.get("supports") or ""}, ensure_ascii=False))
    narration = [str(b) for b in slide.get("approved_narration") or [] if str(b).strip()]
    if narration:
        lines.append("- approved narration: " + " ".join(narration)[:1500])
    return "\n".join(lines) or "- (no sources recorded for this slide)"


def attach_approved_sources(slides: list[dict[str, Any]], blueprint: dict[str, Any] | None) -> None:
    """Copy each blueprint slide's sources, 'supports' lines and narration
    onto the matching deck-contract slide, so authors and the critic judge a
    figure against the evidence the operator approved."""

    if not blueprint:
        return
    metadata = {str(m.get("label")): m for m in (blueprint.get("provenance") or {}).get("source_metadata", []) if isinstance(m, dict)}
    by_id = {str(s.get("slide_id")): s for s in blueprint.get("slides") or [] if isinstance(s, dict)}
    for slide in slides:
        source = by_id.get(str(slide.get("slide_id") or ""))
        if not source:
            continue
        plans = {_source_key(p): p for p in source.get("source_plan") or [] if isinstance(p, dict)}
        slide["approved_sources"] = []
        for mention in source.get("speaker_author_work_mentions") or []:
            if not isinstance(mention, dict):
                continue
            row = _merge_source_fields(metadata.get(_source_key(mention), {}),
                                       plans.get(_source_key(mention), {}), mention)
            slide["approved_sources"].append({**source_record(row), "supports": row.get("supports") or ""})
        slide["approved_narration"] = list(source.get("narration_beats") or [])


def _revision_candidate(
    revised_prompt: str, *, ids: dict[str, str], model: str, record: dict[str, Any],
    slide: dict[str, Any],
) -> dict[str, Any]:
    """Author a revision that validates and compiles, feeding one refusal back.

    A revision that fixed the reviewer's finding but did not fit the canvas
    was discarded outright, finding and all; on ICML 2026 decks 01/02 that
    threw away 29 of 79 rewrites and left 10 figures with no successful
    rewrite in three rounds. The initial authoring path already feeds a
    compiler refusal back once; this gives the revision the same chance.
    """

    def _attempt(prompt: str) -> tuple[dict[str, Any] | None, str]:
        candidate = _extract_json(_call_model(prompt, model=model, caller=__file__))
        candidate.update(ids)
        try:
            spec = load_spec(candidate, strict=False)
        except FigureSpecError as error:
            return None, str(error)
        errors = validate_spec(spec, mode="compile") + _slide_title_errors(spec, slide)
        if errors:
            return None, "; ".join(errors[:6])
        try:
            compile_figure(candidate)
        except FigureCompilerError as error:
            return None, str(error)
        return candidate, ""

    candidate, refusal = _attempt(revised_prompt)
    if candidate is not None:
        return candidate
    record["revision_feedback"] = refusal[:400]
    candidate, second = _attempt(
        f"{revised_prompt}\n\nYOUR REVISION WAS REFUSED BEFORE IT COULD BE DRAWN:\n{refusal[:900]}\n"
        "Keep every change the reviewer asked for and fix exactly these problems. "
        "Return the corrected JSON object only."
    )
    if candidate is None:
        raise FigureCompilerError(f"revision refused twice: {second[:300]}")
    return candidate


class CriticBudgetExceeded(RuntimeError):
    """The critic's worst-case cost exceeds its cap; halt before spending."""


def critic_evidence(slide: dict[str, Any], ledger_hits: list[Any]) -> str:
    """What the figure may state: the slide's sources and its numeric rows."""

    lines: list[str] = [approved_source_lines(slide)]
    for claim in ledger_hits:
        lines.append(
            f"- number {claim.claim_id}: {claim.value}{claim.unit} ({claim.metric or 'unlabelled metric'}) "
            f"-- \"{claim.span[:180]}\" [source {claim.source_id}]"
        )
    return "\n".join(lines) or "- (no sources or numeric rows recorded for this slide)"


def build_critic_prompt(
    *, payload: dict[str, Any], slide: dict[str, Any], archetype: str, evidence: str = "",
) -> str:
    """Ask whether this spec's shape says what the slide's claim says.

    Reads the SPEC, never the pixels. Eleven deterministic rules already refuse
    the structures that can be checked mechanically; what is left is the
    judgement -- would a reader draw a conclusion the claim does not support?
    Asking it here costs a third of a vision call and arrives while the author
    can still act on it, rather than after a deck has been rendered and judged.
    """

    definition = ARCHETYPES[archetype]
    catalog = "\n".join(
        f"     - {name}: {ARCHETYPES[name].summary}" for name in sorted(ARCHETYPES)
        if name != archetype and ARCHETYPES[name].family not in QUOTA_EXEMPT_FAMILIES
    )
    return f"""You are an ADVERSARIAL reviewer of a FIGURE SPEC before it is compiled.
Assume the figure contains at least one mistake and try to find it. You cannot
see a picture. Decide whether the STRUCTURE says what the slide claims, whether
every label and number is supported by the evidence below, and what a reader
would wrongly conclude from it.

THE SLIDE
  title: {slide.get('title') or ''}
  claims: {json.dumps((slide.get('claims') or [])[:3])}
  bullets the reader sees: {json.dumps((slide.get('visible_text') or [])[:5])}

WHAT THE FIGURE MAY STATE (the slide's sources and measured numbers)
{approved_source_lines(slide)}
{evidence or "- (not provided)"}

THE SPEC
  archetype: {archetype} -- {definition.summary}
{json.dumps(payload, indent=2, sort_keys=True)[:5000]}

ASK, IN THIS ORDER
1. Does the shape encode the RELATION the claim states? A sequence claims one
   thing follows another; a branch claims alternatives; a cycle claims return.
   For `flow`, inspect each edge's `kind`: `flow` means a causal or temporal
   step, `parallel` means complementary methods without ordering, and
   `feedback` means a return edge. A claimed loop needs both its forward path
   and the return edge. A claimed merge needs a named `merge` node with at
   least two incoming branches; it can itself be the result.
2. What would a reader wrongly conclude? Name the specific misreading.
3. Is anything joined to nothing, or named but never shown?
4. Do the axes, points or panels measure or depict one comparable thing?
5. Does the emphasis fall on the subject of the claim?
6. Is this the archetype that says the relation best? A picture archetype (a
   3-D scene, a frame strip) shows a subject and says relations poorly; a
   labelled pipeline drawn as a scene, or alternatives drawn as a sequence,
   is `wrong_archetype`. If so, NAME the archetype to use in suggested_change,
   from this catalog:
{catalog}

7. FIDELITY. Judge overclaim and misattribution against per-source
   contribution_note and work_kind, never against the slide claim, supports
   sentence or narration. Check every outcome node and every cross-source edge.
   Joining different sources' methods by edges, nesting or sequence without
   synthesis_label is `composed_sources`; a labelled synthesis is advisory.
   Showing a position paper's argument as a result is `position_as_result`. Every number in the spec must match a numeric row above (value,
   unit, metric and condition) -- otherwise `unsupported_number`. A matrix cell
   marked yes/no/partial that no source states for that row, and a plotted
   point whose `basis` (or measured row) no source states, are `unsupported_number`
   too. Every label
   naming a system, method, benchmark, work or person must appear in the slide
   text, the sources, their "supports" lines or the approved narration above --
   otherwise `unsupported_label`. A result credited to
   the wrong work or person is `misattribution`. A figure that asserts more
   than its bound source notes (causation for correlation, a general law from one
   result, a ranking the sources do not make) is `overclaim`.

SEVERITY. `major`: a reader would take away something false, unsupported or
misattributed, or the figure fails to show the claim. `minor`: clarity or
emphasis that leaves the reader's conclusion correct. Only report real issues;
do not invent minor ones to have something to say.

Do NOT comment on layout, colour, spacing, overlap or fonts. A deterministic
compiler owns all of that and a separate gate checks it.

Return ONLY a JSON object:
{{"verdict": "accept" | "revise",
  "issues": [{{"code": one of {list(CRITIC_CODES)},
              "severity": "major" | "minor",
              "detail": "the misreading, in one sentence",
              "suggested_change": "the concrete change to the spec"}}]}}
Use "revise" only when there is at least one major issue."""


def named_archetype_switch(critique: dict[str, Any], *, current: str, visual_role: str = "") -> str:
    """The archetype a critic's fix names, if the switch is sound.

    The critic could say `wrong_archetype` from the first rubric on, and the
    re-author then regenerated the spec under the SAME archetype, because the
    selector had fixed it before the model ever saw the slide -- the one
    finding the revision could not act on. A named, known archetype other than
    the current one is honoured, unless it is a picture the brief is too
    relation-heavy for, which is the mistake the selector just avoided.

    Restricted to `wrong_archetype`-coded issues once. In practice the model
    names the real fix -- "restructure as a decision_tree", "use flow
    archetype" -- under whichever of the rubric's OTHER codes the mismatch
    happened to read as (an alternatives-drawn-as-sequence claim reads as
    `alternatives_as_sequence`, a claim the shape never encodes reads as
    `claim_not_encoded`), not always `wrong_archetype` itself. Restricting the
    switch to one code meant the reauthor rebuilt the SAME wrong archetype
    with the critic's prose pasted in as advice it had no shape to act on:
    two real decks each carried one figure -- a branch-then-merge recovery
    forced into a `loop`'s ring, a linear diagnostic chain forced into the
    same ring -- that a critic correctly named the fix for and the pipeline
    never applied. Matched against `suggested_change` alone (the concrete
    instruction), not `detail` (freeform prose more likely to use an
    archetype's name, like "flow", as an ordinary English word rather than a
    direction to switch).

    A name only counts near a switch verb ("use", "switch", "replace ...
    with", ...), not merely present anywhere in the sentence. `flow` and
    `loop` are also ordinary English words and `flow` is also a generic edge
    `kind` -- one real spec's fix was "Switch to layered_stack ... to show
    the corrective loop" (a `loop` 150 characters past the verb, describing
    the return path `layered_stack` already draws, not naming an archetype)
    and another's was "Add explicit edge kinds (e.g. flow from X to Y)" (an
    edge-kind example, no switch verb anywhere). Unwindowed matching took
    both as a named switch to `loop` / `flow` and broke a figure the critic
    never asked to change archetype at all. The window is wide enough for
    "replace <a clause> with <name>" (measured at 56 characters on a real
    critique) and well short of a same-sentence but unrelated mention (150).
    """

    switch_verb = re.compile(
        r"\b(use|switch|replace|restructure|author it|draw it|recast|change|redo|rebuild)\b"
    )
    for issue in critique.get("issues") or []:
        text = str(issue.get("suggested_change") or "").lower()
        for name in sorted(ARCHETYPES, key=len, reverse=True):
            # The slide-text layout is the operator's outcome (patch_figure_spec.py
            # --text-layout), never a model's way out of drawing the figure.
            if name == current or ARCHETYPES[name].family in QUOTA_EXEMPT_FAMILIES:
                continue
            match = re.search(rf"\b{re.escape(name)}\b", text)
            if match is None:
                continue
            verbs = list(switch_verb.finditer(text, 0, match.start()))
            if not verbs or match.start() - verbs[-1].end() > 70:
                continue
            if name in PICTURE_ARCHETYPES and relation_heavy(visual_role, name):
                continue
            return name
    return ""


def _collection_sizes(node: Any, path: str = "content",
                      out: dict[str, int] | None = None) -> dict[str, int]:
    """How many items sit in every list the content tree holds, by key path."""

    sizes: dict[str, int] = {} if out is None else out
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, list):
                sizes[f"{path}.{key}"] = len(value)
            _collection_sizes(value, f"{path}.{key}", sizes)
    elif isinstance(node, list):
        for value in node:
            _collection_sizes(value, f"{path}[]", sizes)
    return sizes


def unrequested_losses(original: Any, revised: Any, critique: dict[str, Any]) -> list[str]:
    """Content a revision deleted that the critique never asked about.

    A critic finding is a pointed request -- move this marker, drop that
    branch, relabel that axis. The re-author, though, regenerates the whole
    spec, so it can honour the request and shed unrelated content in the same
    breath. That loss passes `validate_spec` and `compile_figure` alike,
    because a smaller figure is a well-formed figure that fits more easily:
    one revision answered a note about two highlighted groups by deleting
    nine of the slide's fourteen abilities, and the deck shipped a tree whose
    own title said fourteen.

    Two conditions, because either alone misfires. Rewording is not loss --
    an axis relabelled from "needs abundant labelled/paired data" to
    "abundant paired data" drops a string while keeping the content -- so
    require a *collection* to have actually shrunk. And a deletion the critic
    asked for is legitimate, so require that the vanished labels go unnamed
    in the critique's own prose. The word match is loose on purpose: the
    caller refuses on any survivor, and the safe error is to accept a removal
    the critic did request.
    """

    shrunk = _collection_sizes(revised)
    if not any(size > shrunk.get(path, 0) for path, size in _collection_sizes(original).items()):
        return []

    prose = " ".join(
        f"{issue.get('code', '')} {issue.get('detail', '')} {issue.get('suggested_change', '')}"
        for issue in critique.get("issues") or []
    ).lower()
    kept = {label.lower() for label in content_labels(revised)}
    return [
        label
        for label in content_labels(original)
        if label.lower() not in kept
        and not any(word in prose for word in re.findall(r"[a-z]{4,}", label.lower()))
    ]


def _critique(
    payload: dict[str, Any], slide: dict[str, Any], archetype: str,
    *, model: str, workspace: Path | None = None, evidence: str = "",
) -> dict[str, Any]:
    # Its own receipt context, so the critic's spend is attributed to the
    # critic's task id rather than to the authoring task that happens to be on
    # the stack. Contexts save and restore, so nesting is safe.
    with provider_receipt_context(
        workspace_root=workspace, stage_id="figure_spec_critic",
        task_id=CRITIC_TASK_ID, caller=__file__,
    ):
        return _critique_call(payload, slide, archetype, model=model, evidence=evidence)


def _critique_call(
    payload: dict[str, Any], slide: dict[str, Any], archetype: str,
    *, model: str, evidence: str = "",
) -> dict[str, Any]:
    raw = _call_model(
        build_critic_prompt(payload=payload, slide=slide, archetype=archetype, evidence=evidence),
        # Room for the model's reasoning AND the answer. At 1200 the reasoning
        # consumed the whole budget on some specs and the call came back with a
        # thinking block and no text at all -- which looked like a provider
        # failure and was really a token budget.
        model=model, caller=__file__, max_tokens=4000, task_id=CRITIC_TASK_ID,
    )
    parsed = _extract_json(raw)
    verdict = str(parsed.get("verdict") or "accept").strip().lower()
    issues = []
    for issue in (parsed.get("issues") or []):
        if not isinstance(issue, dict):
            continue
        code = str(issue.get("code") or "other").strip()
        severity = str(issue.get("severity") or "major").strip().lower()
        issues.append({
            "code": code if code in CRITIC_CODES else "other",
            # An unlabelled issue counts as major: a reviewer that forgot the
            # field must not have its finding silently waved through.
            "severity": severity if severity in CRITIC_SEVERITIES else "major",
            "detail": str(issue.get("detail") or "").strip(),
            "suggested_change": str(issue.get("suggested_change") or "").strip(),
        })
    has_major = any(issue["severity"] == "major" for issue in issues)
    return {
        "verdict": "revise" if verdict == "revise" and has_major else "accept",
        "issues": issues,
        "rubric_version": CRITIC_RUBRIC_VERSION,
        "model_task_id": CRITIC_TASK_ID,
    }


def fallback_spec(*, slide: dict[str, Any], archetype: str, figure_id: str) -> dict[str, Any]:
    """A minimal valid spec, used by --dry-run and when a response cannot be repaired.

    It is deliberately thin: enough to compile and to prove the wiring, never
    enough to pass for authored content.
    """

    title = str(slide.get("display_title") or slide.get("title") or slide.get("slide_id") or "")
    claim = str(slide.get("viewer_facing_claim") or slide.get("visual_takeaway") or title)
    entities = _required_entities(slide)[:4]
    # A flow needs at least one edge, so the placeholder always has at least two
    # stages: the slide's own entities first, then generic stages to fill.
    labels = list(entities)
    for filler in ("Input", "Method", "Result"):
        if len(labels) >= max(2, len(entities) + 1):
            break
        if filler not in labels:
            labels.append(filler)
    nodes = [
        {"id": f"n{index}", "label": label, "kind": "process"}
        for index, label in enumerate(labels, start=1)
    ]
    return {
        "spec_version": SPEC_VERSION,
        "figure_id": figure_id,
        "slide_id": str(slide.get("slide_id") or ""),
        "section": str(slide.get("section_id") or ""),
        "archetype": "flow" if archetype not in {"flow"} else archetype,
        "title": title[:70],
        "claim": claim[:180] or title[:180] or "This slide's figure.",
        "takeaway": " ".join((claim or title).split()[:8]) or "See the slide",
        "grounding": {"kind": "schematic", "note": PLACEHOLDER_NOTE},
        "required_entities": entities,
        "content": {
            "nodes": nodes,
            "edges": [
                {"from": nodes[index]["id"], "to": nodes[index + 1]["id"]}
                for index in range(len(nodes) - 1)
            ],
        },
    }


def _extract_json(text: str) -> dict[str, Any]:
    body = str(text or "").strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", body, re.DOTALL)
    if fence:
        body = fence.group(1)
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        # Say what DID come back; "no JSON object" alone gives a reader nothing
        # to act on, and an empty body means something different from prose.
        preview = body[:160].replace("\n", " ") or "(empty response)"
        raise ValueError(f"no JSON object in the response; got: {preview}")
    payload = json.loads(body[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("the response JSON is not an object")
    return payload


TRANSIENT_ATTEMPTS = 4


def _with_backoff(call: Any) -> Any:
    """Retry a provider call whose failure is transient.

    Authoring slides concurrently means several calls are in flight at once,
    so a rate limit is an ordinary event rather than an emergency. It was not
    survivable before: the only try/except around a call caught JSON errors,
    so a 429 propagated out of ``generate`` and abandoned the whole deck
    mid-run. Bounded, with jitter, and anything that is not transient is
    re-raised immediately.
    """

    transient = ("rate_limit", "overloaded", "429", "529", "timeout", "connection")
    for attempt in range(TRANSIENT_ATTEMPTS):
        try:
            return call()
        except TypeError:
            raise
        except Exception as error:  # noqa: BLE001 - classified by message, then re-raised
            text = f"{type(error).__name__} {error}".lower()
            if attempt == TRANSIENT_ATTEMPTS - 1 or not any(word in text for word in transient):
                raise
            time.sleep(min(30.0, 2.0 ** attempt) + random.uniform(0.0, 1.0))
    raise RuntimeError("unreachable")


def _slide_title_errors(spec: Any, slide: dict[str, Any]) -> list[str]:
    """Repair instructions when the figure heading only repeats the slide's title."""

    title = str(slide.get("display_title") or slide.get("title") or "")
    return [
        f"{finding.code}: {finding.message} -- {finding.suggested_change}"
        for finding in (title_repeats_slide_title(spec, title) + unsupported_outcomes(spec, slide) + source_fidelity_errors(spec, slide))
    ]


def _call_model(prompt: str, *, model: str, caller: str, max_tokens: int = 8000,
                task_id: str = TASK_ID) -> str:
    """One authoring, critic or repair call, with extended thinking OFF.

    The provider is whichever the governed task allows for ``model`` (Claude,
    Gemini, OpenAI or an OpenRouter model such as Kimi): ``complete`` resolves
    it from the registry and refuses a model the task has not allowed.

    Measured on deck 05: authoring calls left at the provider's default spent
    their whole budget inside a thinking block and returned no text at all --
    8000 output tokens, `stop_reason=max_tokens`, zero characters, 90 seconds,
    $0.13. The repair loop then fed the unparseable empty string back and the
    retry did the same, so one slide could burn three attempts and four
    minutes and still be recorded as blocked. Raising the ceiling does not
    help: the same prompt at 16000 tokens thinks for 179 seconds and still
    returns nothing. The same prompt with thinking disabled answers in 11
    seconds with 991 tokens. 13 of 236 authoring calls since 2026-09-08 ended
    this way, and the ones that did return paid roughly 2.3x the wall clock
    for thinking a structured-JSON task does not need -- the reasoning that
    matters here is the spec critic's, which is a separate call with its own
    rubric. ``thinking=False`` is how every provider is asked for that.
    """

    result = _with_backoff(
        lambda: complete(
            task_id,
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,  # a feature_matrix or heatmap spec can exceed 4k output tokens
            thinking=False,
            caller=caller,
        )
    )
    return result.text


def generate(
    *,
    deck_dir: Path,
    workspace: Path,
    dry_run: bool = False,
    slide_limit: int = 0,
    model: str = "",
    skip_existing: bool = False,
    slide_ids: tuple[str, ...] = (),
    spec_dir: Path | None = None,
    critic: bool = True,
    critic_budget_usd: float = 6.0,
    critic_reauthor: bool = True,
    critic_runner: Any = None,
    preferred: dict[str, str] | None = None,
    workers: int = 6,
    provider: str = "",
    critic_model: str = "",
    critic_provider: str = "",
) -> dict[str, Any]:
    preferred_archetypes = dict(preferred or {})
    unknown = sorted(set(preferred_archetypes.values()) - set(ARCHETYPES))
    if unknown:
        raise SystemExit(f"--prefer names archetypes that do not exist: {', '.join(unknown)}")
    deck_contract = _load_json(deck_dir / "presentation_deck.json")
    slides = _slide_rows(deck_contract)
    blueprint_path = deck_dir / "presentation_blueprint.json"
    attach_approved_sources(slides, _load_json(blueprint_path) if blueprint_path.is_file() else None)
    binding_path = deck_dir / "source_binding_report.json"
    if binding_path.is_file():
        bindings = {str(row.get("slide_id")): row for row in _load_json(binding_path).get("slides", [])}
        for slide in slides:
            bound = bindings.get(str(slide.get("slide_id")), {}).get("visible_references")
            if bound:
                approved = {_source_key(row): row for row in slide.get("approved_sources") or []}
                merged = []
                for row in bound:
                    prior = approved.get(_source_key(row), {})
                    if (prior.get("source_id") and row.get("source_id")
                            and prior["source_id"] != row["source_id"]):
                        prior = {}  # A rebound label cannot inherit another work's note.
                    combined = _merge_source_fields(prior, row)
                    merged.append({**source_record(combined), "supports": combined.get("supports") or ""})
                slide["approved_sources"] = merged

    if slide_ids:
        # Substring match, so a caller can name a slide by its readable prefix
        # rather than by the full generated id.
        wanted = tuple(str(value).strip() for value in slide_ids if str(value).strip())
        slides = [
            slide for slide in slides
            if any(token in str(slide.get("slide_id") or "") for token in wanted)
        ]
        missing = [
            token for token in wanted
            if not any(token in str(slide.get("slide_id") or "") for slide in slides)
        ]
        if missing:
            raise SystemExit(f"no slide matches {missing} in {deck_dir}")
    if slide_limit:
        slides = slides[:slide_limit]

    ledger_path = deck_dir / LEDGER_NAME
    if ledger_path.is_file():
        ledger = load_ledger(ledger_path)
    else:
        ledger = build_ledger(
            topic_id=deck_dir.name,
            source_ledger=_load_json(deck_dir / "speaker_author_work_ledger.json"),
        )
        ledger.write(ledger_path)

    spec_dir = spec_dir or (deck_dir / SPEC_DIR_NAME)
    spec_dir.mkdir(parents=True, exist_ok=True)
    resolved_provider, resolved_model = resolve_selection(TASK_ID, provider or None, model or None)
    critic_provider, critic_model = resolve_selection(
        CRITIC_TASK_ID, critic_provider or None, critic_model or None
    )
    if not dry_run:
        needed = {resolved_provider} | ({critic_provider} if critic_runner is None else set())
        for needed_provider in sorted(needed):
            if not has_credentials(needed_provider):
                raise SystemExit(
                    f"{' or '.join(credential_names(needed_provider) or ())} is required for "
                    f"{needed_provider} (source the repo-local .env) or pass --dry-run"
                )

    # Seed the rotation history from the specs that are NOT being re-authored.
    # Without it a targeted run sees an empty deck and rotates against nothing,
    # so ten re-authored slides can all land on the same archetype while the
    # thirty-nine around them already use it.
    run_critic = bool(critic) and (critic_runner is not None or not dry_run)
    history: list[str] = []
    if slide_ids:
        selected_ids = {str(slide.get("slide_id") or "") for slide in slides}
        for existing in sorted(spec_dir.glob("*.json")) if spec_dir.is_dir() else []:
            try:
                payload = json.loads(existing.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if str(payload.get("slide_id") or "") not in selected_ids:
                history.append(str(payload.get("archetype") or ""))
    written: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    accepted_path = deck_dir / ACCEPTED_DIGESTS_NAME
    try:
        accepted_digests: dict[str, str] = json.loads(accepted_path.read_text(encoding="utf-8")) if accepted_path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        accepted_digests = {}

    def _author_one(
        *, slide: dict[str, Any], slide_id: str, section_id: str, figure_id: str,
        hits: list[Any], outcome: Any, use_critic: bool,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Author one slide's spec: the row for the receipt, or a blocker.

        Everything this needs is passed in, because it runs on a worker thread
        alongside the other slides. It writes only files named after its own
        figure, and returns rather than appending, so the receipt is assembled
        in slide order whatever order the threads finish in.
        """

        prompt = build_prompt(slide=slide, archetype=outcome.archetype, ledger=ledger, ledger_hits=hits)
        (spec_dir / f"{figure_id}.prompt.txt").write_text(prompt, encoding="utf-8")
        brief_role = parse_visual_role(str(slide.get("visual_role") or ""))
        _kept_brief, dropped_brief = filter_components(brief_role["components"], slide, depiction=brief_role["prefix"])

        payload: dict[str, Any] | None = None
        errors: list[str] = []
        reused_existing = False
        carried_lint: list[str] = []
        spec_path_existing = spec_dir / f"{figure_id}.json"
        if skip_existing and spec_path_existing.is_file():
            # Resume without paying twice: a spec already on disk that still
            # validates is kept as authored. Only the slides that failed last
            # time reach the model again.
            try:
                existing = json.loads(spec_path_existing.read_text(encoding="utf-8"))
                note = str((existing.get("grounding") or {}).get("note") or "")
                # A placeholder validates cleanly -- that is the whole danger. It
                # must never satisfy a resume, or a --dry-run preflight silently
                # becomes the deck's real content.
                existing_spec = load_spec(existing, strict=False)
                standing = spec_standing(spec_dir, figure_id, spec_digest(existing), accepted_digests)
                lint_errors = _slide_title_errors(existing_spec, slide)
                if (lint_errors and standing == "operator_patched"
                        and PLACEHOLDER_NOTE not in note and not validate_spec(existing_spec)):
                    # Re-authoring would silently discard the operator's edit:
                    # the operator fixes it instead.
                    return None, {
                        "slide_id": slide_id, "figure_id": figure_id, "code": "operator_patch_fails_lint",
                        "archetype": str(existing.get("archetype") or ""), "errors": list(lint_errors),
                    }
                # A spec that already passed (or that the operator accepted) is not
                # re-opened by a lint added since: re-authoring it is a fresh model
                # call that changes content nobody asked to change. Its lint
                # findings are recorded on the receipt row instead.
                carried_lint = list(lint_errors) if (lint_errors and standing == "accepted") else []
                if (PLACEHOLDER_NOTE not in note
                        and not validate_spec(existing_spec)
                        and (not lint_errors or carried_lint)):
                    # A structurally valid spec can still fail measured layout
                    # or lose a label. Reusing it on resume would turn the
                    # previous compiler refusal into a permanent blocker.
                    compile_figure(existing)
                    payload = existing
                    reused_existing = True
            except (OSError, json.JSONDecodeError, FigureCompilerError):
                payload = None
        if payload is not None:
            pass
        elif dry_run:
            payload = fallback_spec(slide=slide, archetype=outcome.archetype, figure_id=figure_id)
        else:
            attempt_prompt = prompt
            last_valid_shape: dict[str, Any] | None = None
            for attempt in range(MAX_REPAIR_ATTEMPTS + 1):
                raw = _call_model(attempt_prompt, model=resolved_model, caller=__file__)
                try:
                    candidate = _extract_json(raw)
                except (ValueError, json.JSONDecodeError) as error:
                    errors = [
                        f"unparseable response: {error}"
                        if raw.strip()
                        # An empty body is the model spending its whole budget
                        # before writing anything, not a malformed spec: say so,
                        # so the receipt names the real problem.
                        else "empty response: the call returned no text within its token budget"
                    ]
                else:
                    candidate.setdefault("spec_version", SPEC_VERSION)
                    candidate["figure_id"] = figure_id
                    candidate["slide_id"] = slide_id
                    # Editorial metadata, stamped from the deck contract rather
                    # than authored: the model has no standing to say which
                    # section a slide belongs to.
                    candidate["section"] = section_id
                    try:
                        spec = load_spec(candidate, strict=False)
                    except FigureSpecError as error:
                        errors = [str(error)]
                    else:
                        errors = validate_spec(spec) + _slide_title_errors(spec, slide)
                        if not errors:
                            payload = candidate
                            break
                        if (not validate_spec(spec, mode="compile")
                                and not _slide_title_errors(spec, slide)):
                            # Only the authoring budget is exceeded; remember it.
                            last_valid_shape = candidate
                if attempt < MAX_REPAIR_ATTEMPTS:
                    attempt_prompt = (
                        f"{prompt}\n\nYOUR PREVIOUS ATTEMPT WAS REJECTED. Fix exactly these problems "
                        f"and return the corrected JSON object only:\n- " + "\n- ".join(errors[:6])
                    )

        budget_overrun: list[str] = []
        if payload is None and last_valid_shape is not None:
            # The label/item budget is a floor measured on a worst-case shape,
            # not a boundary (see figures.spec.budget_findings): a spec over it
            # may still draw. After the repairs, let the compiler decide --
            # ICML 2026 deck 01 lost five region figures to labels 1-6
            # characters over. A spec that compiles is kept and the overrun
            # recorded; one that does not stays a blocker.
            try:
                compile_figure(last_valid_shape)
            except FigureCompilerError:
                pass
            else:
                payload = last_valid_shape
                budget_overrun = errors[:6]
        if payload is None:
            return None, {
                "code": "figure_spec_invalid",
                "slide_id": slide_id,
                "archetype": outcome.archetype,
                "errors": errors[:6],
            }

        # The adversarial critic reads the spec before it is compiled, at most
        # MAX_CRITIC_ROUNDS times. A major issue buys a re-author and the
        # revision is read again; a revision that is not clean is rejected and
        # the current spec stands. Minor issues are recorded, never acted on.
        critique: dict[str, Any] | None = None
        sidecar = spec_dir / f"{figure_id}.critic.json"
        # A spec reused on resume already had its critique; paying again
        # for an unchanged spec is what --skip-existing exists to avoid. Its
        # recorded critique is carried into this receipt, or a resumed run
        # reports no open findings for figures that still have them.
        if reused_existing and sidecar.is_file():
            try:
                critique = {**json.loads(sidecar.read_text(encoding="utf-8")), "reused": True}
            except (OSError, json.JSONDecodeError):
                critique = None
        if use_critic and not reused_existing:
            evidence = critic_evidence(slide, hits)
            rounds: list[dict[str, Any]] = []
            pending: dict[str, Any] | None = None
            for round_number in range(1, MAX_CRITIC_ROUNDS + 1):
                if pending is None:
                    archetype_now = str(payload.get("archetype") or outcome.archetype)
                    try:
                        critique = (
                            critic_runner(payload, slide, archetype_now)
                            if critic_runner is not None
                            else _critique(payload, slide, archetype_now, model=critic_model,
                                           workspace=workspace, evidence=evidence)
                        )
                    except Exception as error:  # noqa: BLE001 - advice must never stop authoring.
                        critique = {"verdict": "accept", "issues": [],
                                    "error": f"{type(error).__name__}: {error}"}
                    critique = dict(critique)
                    critique["round"] = round_number
                    rounds.append(critique)
                else:
                    # The last re-author failed; retry it against the same
                    # findings rather than paying to read an unchanged spec.
                    critique = pending
                    pending = None
                major = [i for i in critique.get("issues") or [] if i.get("severity", "major") == "major"]
                if critique.get("verdict") != "revise" or not major:
                    break
                if round_number == MAX_CRITIC_ROUNDS or not critic_reauthor or dry_run:
                    break
                notes = "\n- ".join(
                    f"{issue['code']}: {issue['detail']} -> {issue['suggested_change']}"
                    for issue in major[:4]
                )
                switched = named_archetype_switch(
                    critique, current=str(payload.get("archetype") or outcome.archetype),
                    visual_role=str(slide.get("visual_role") or ""),
                )
                base_prompt = (
                    build_prompt(slide=slide, archetype=switched, ledger=ledger, ledger_hits=hits)
                    if switched else prompt
                )
                revised_prompt = (
                    f"{base_prompt}\n\nYOUR CURRENT SPEC:\n{json.dumps(payload, sort_keys=True)[:4000]}\n\n"
                    f"AN ADVERSARIAL REVIEWER FOUND THESE MAJOR ISSUES:\n- {notes}\n"
                    + ("The reviewer asked for a different archetype; the instructions above "
                       "now name it, so author the figure under that archetype.\n" if switched else "")
                    + "Fix exactly these issues, keep everything else, and return the corrected JSON object only."
                )
                try:
                    candidate = _revision_candidate(
                        revised_prompt,
                        ids={"spec_version": SPEC_VERSION, "figure_id": figure_id,
                             "slide_id": slide_id, "section": section_id},
                        model=resolved_model, record=critique, slide=slide,
                    )
                    # A different archetype has different content keys; the
                    # loss check compares like with like and is skipped here.
                    lost = [] if switched else unrequested_losses(
                        payload.get("content"), candidate.get("content"), critique
                    )
                    if lost:
                        # Content the critic never mentioned, deleted anyway.
                        critique["reauthored"] = False
                        critique["revision_rejected"] = True
                        critique["unrequested_losses"] = lost
                    else:
                        payload = candidate
                        critique["reauthored"] = True
                        if switched:
                            critique["archetype_switched"] = switched
                except Exception as error:  # noqa: BLE001
                    critique["reauthored"] = False
                    critique["revision_error"] = f"{type(error).__name__}: {error}"
                if not critique.get("reauthored"):
                    pending = critique
            if critique is not None:
                final_major = [
                    i for i in critique.get("issues") or []
                    if i.get("severity", "major") == "major"
                ] if critique.get("verdict") == "revise" and not critique.get("reauthored") else []
                critique = {
                    **critique,
                    "rounds": rounds,
                    "critic_calls": len(rounds),
                    "reauthored": any(r.get("reauthored") for r in rounds),
                    **({"archetype_switched": switched_to[-1]} if (switched_to := [
                        str(r["archetype_switched"]) for r in rounds if r.get("archetype_switched")
                    ]) else {}),
                    "unresolved_major": final_major,
                }
                sidecar.write_text(json.dumps(critique, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        # The compiler's own refusal, fed back once while the author is still
        # in the loop. Four of deck 04's twenty-five first drafts validated and
        # then refused at compile -- a label that read as a cut-off sentence, a
        # group with more modules than a column holds, two labels overlapping
        # by 4px -- each a one-line change the model makes in a $0.05 call and
        # an operator otherwise makes by hand after the whole deck has run.
        compile_feedback: dict[str, Any] | None = None
        if not dry_run:
            try:
                compile_figure(payload)
            except FigureCompilerError as error:
                compile_feedback = {"first_refusal": str(error)[:700], "repaired": False}
                # "carries content the layout never drew" means the archetype
                # is the problem, and telling the author to keep it is telling
                # it to fail again -- which is what happened to deck 05's
                # slide 25: the same refusal twice, the second time for having
                # obeyed. Every other refusal is a wording or a count.
                disagrees = "the spec and the archetype disagree" in str(error)
                instruction = (
                    "Choose an archetype that READS the content you want to show, and rewrite "
                    "the content under that archetype's keys -- or keep this one and drop the "
                    "content it cannot draw. Do not return the same spec again."
                    if disagrees else
                    "Keep the same archetype and every entity that must appear. Reword a label "
                    "that reads as a cut-off sentence, shorten labels that overlap, or carry "
                    "fewer items where the figure does not fit."
                )
                repair_prompt = (
                    f"{prompt}\n\nTHE COMPILER REFUSED YOUR SPEC:\n{str(error)[:900]}\n"
                    f"{instruction} Return the corrected JSON object only."
                )
                try:
                    candidate = _extract_json(
                        _call_model(repair_prompt, model=resolved_model, caller=__file__)
                    )
                    candidate.update({"spec_version": SPEC_VERSION, "figure_id": figure_id,
                                      "slide_id": slide_id, "section": section_id})
                    revised = load_spec(candidate, strict=False)
                    repair_errors = validate_spec(revised) + _slide_title_errors(revised, slide)
                    if not repair_errors:
                        compile_figure(candidate)
                        payload = candidate
                        compile_feedback["repaired"] = True
                    else:
                        compile_feedback["repair_invalid"] = repair_errors[:3]
                except Exception as error:  # noqa: BLE001 - the original stands; compile names it
                    compile_feedback["repair_error"] = f"{type(error).__name__}: {error}"[:300]

        if compile_feedback is not None and not compile_feedback.get("repaired"):
            # The spec is written either way, so an operator can read it -- but
            # a spec the compiler refuses is not a pass, and saying so here
            # names the slide while the author is still in the loop rather than
            # letting the deck compile discover it two steps later.
            _write_spec_preserving_refusal(spec_dir / f"{figure_id}.json", payload)
            return None, {
                "code": "figure_spec_refused_by_compiler",
                "slide_id": slide_id,
                "archetype": str(payload.get("archetype") or outcome.archetype),
                "errors": [str(compile_feedback.get("first_refusal") or "")[:400]],
            }

        spec_path = spec_dir / f"{figure_id}.json"
        _write_spec_preserving_refusal(spec_path, payload)
        return {
            "slide_id": slide_id,
            "section_id": section_id,
            "figure_id": figure_id,
            "spec_sha256": spec_digest(payload),
            **({"lint_findings_on_accepted_spec": carried_lint} if reused_existing and carried_lint else {}),
            "archetype": payload.get("archetype"),
            "selection": outcome.to_dict(),
            "ledger_rows_offered": [claim.claim_id for claim in hits],
            "critic": critique,
            "compile_feedback": compile_feedback,
            **({"brief_components_dropped": dropped_brief} if dropped_brief else {}),
            **({"budget_overrun_compiled": budget_overrun} if budget_overrun else {}),
            "spec_path": str(
                spec_path.relative_to(deck_dir)
                if spec_path.is_relative_to(deck_dir) else spec_path
            ),
        }, None

    # Phase 1: choose every slide's archetype, in order and for free. Rotation
    # and the family-share rule both read the choices already made, so this
    # pass is sequential -- and it costs nothing, because selection makes no
    # provider call. Fixing the plan here is also what lets the paid work fan
    # out: the history a slide rotates against no longer depends on when some
    # other slide's call happens to return.
    plans: list[dict[str, Any]] = []
    for slide in slides:
        slide_id = str(slide.get("slide_id") or "")
        section_id = str(slide.get("section_id") or "")
        figure_id = f"{slide_id}-figure"
        beats = " ".join(str(item) for item in (slide.get("narration_beats") or slide.get("narration_hooks") or []))
        query = " ".join(filter(None, [str(slide.get("display_title") or slide.get("title") or ""), beats]))
        # A number may only ground the slide that cites its source. The search
        # is keyword-based, and on a single-topic deck the keywords match
        # everywhere: without this the author was offered another paper's
        # measurements and the demotion below was decided by them.
        own_sources = [str(value) for value in (slide.get("source_ids") or [])]
        hits = ledger.search(query, sources=own_sources)
        # Offering a weakly-matching row to the prompt is harmless -- the author
        # can ignore it. Deciding "this slide HAS data" on that same weak match
        # is not: it suppresses demotion, and the model is then asked for a
        # scatter plot of numbers that do not exist. Demote on the strong bar.
        strong_hits = ledger.search(query, min_score=ledger.STRONG_MATCH_SCORE,
                                    sources=own_sources)
        outcome = select_archetype(
            slide_id=slide_id,
            # An operator override, for the one case the cues cannot reach: the
            # per-section quota wants a section to SHOW a number it already
            # cites, and the selector reads the figure brief, which describes
            # the mechanism rather than the measurement.
            preferred=next((value for key, value in preferred_archetypes.items()
                            if key in slide_id), ""),
            visual_role=str(slide.get("visual_role") or ""),
            # The blueprint's figure brief says what the figure is ABOUT; without
            # it every "mechanism_pipeline" slide -- video, 3-D, image operation
            # alike -- collapses onto the same box-and-arrow diagram.
            brief=str(slide.get("visual_figure_brief") or slide.get("visual_narration_hook") or ""),
            has_ledger_rows=bool(strong_hits),
            ledger_rows=len(strong_hits),
            history=history,
        )
        history.append(outcome.archetype)
        plans.append({
            "slide": slide, "slide_id": slide_id, "section_id": section_id,
            "figure_id": figure_id, "hits": hits, "outcome": outcome,
        })

    # The critic is always on for every figure. Its worst case (every slide,
    # every round) must fit the cap before anything is spent; a deck that does
    # not fit halts for approval instead of quietly skipping later slides.
    if run_critic and critic_runner is None:
        worst_case = len(plans) * MAX_CRITIC_ROUNDS * CRITIC_EST_USD_PER_CALL
        if worst_case > critic_budget_usd:
            raise CriticBudgetExceeded(
                f"adversarial figure critic worst case ${worst_case:.2f} for {len(plans)} slides x "
                f"{MAX_CRITIC_ROUNDS} rounds exceeds --critic-budget-usd ${critic_budget_usd:.2f}; "
                "raise the cap explicitly to proceed"
            )
    for plan in plans:
        plan["use_critic"] = run_critic

    # Phase 2: author them. One slide's spec never depends on another's and
    # every call is network-bound, so they go out together -- 25 slides at
    # ~25s each is ten minutes in a row and under two when they overlap.
    results: list[tuple[dict[str, Any] | None, dict[str, Any] | None]] = [(None, None)] * len(plans)
    if workers > 1 and len(plans) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            # Each worker gets its OWN copy of the caller's context. Provider
            # receipts resolve their run and stage from a ContextVar first and
            # only fall back to process-wide environment variables, and a
            # worker thread inherits neither by default -- so without this a
            # concurrent critic call (which opens its own context, and sets
            # those same environment variables for the whole process) would
            # file another thread's authoring receipt under `figure_spec_critic`.
            # One copy per task, because a Context cannot be entered twice at once.
            pending = {
                pool.submit(contextvars.copy_context().run, _author_one, **plan): index
                for index, plan in enumerate(plans)
            }
            for future in concurrent.futures.as_completed(pending):
                results[pending[future]] = future.result()
    else:
        for index, plan in enumerate(plans):
            results[index] = _author_one(**plan)

    for row, blocker in results:
        if row is not None:
            written.append(row)
        if blocker is not None:
            blockers.append(blocker)
    if not dry_run and written:
        # Every written spec passed on its own; record it so a later lint does
        # not re-open it. Per slide, so one blocked slide never un-accepts the rest.
        accepted_digests.update({row["figure_id"]: row["spec_sha256"] for row in written})
        accepted_path.write_text(json.dumps(accepted_digests, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    critic_calls = sum(
        int((row.get("critic") or {}).get("critic_calls") or 1)
        for row, _ in results
        if row and row.get("critic") and not row["critic"].get("reused")
    )


    return {
        "receipt_type": "conference_pipeline.figures.spec_authoring",
        "task_id": TASK_ID,
        "provider": "dry-run" if dry_run else resolved_provider,
        "model": "dry-run" if dry_run else resolved_model,
        "deck_dir": str(deck_dir),
        "workspace": str(workspace),
        "dry_run": dry_run,
        "slide_count": len(slides),
        "spec_count": len(written),
        "numeric_ledger_claims": len(ledger.claims),
        "specs": written,
        "critic": {
            "enabled": bool(critic),
            "task_id": CRITIC_TASK_ID,
            "provider": critic_provider,
            "model": critic_model,
            "rubric_version": CRITIC_RUBRIC_VERSION,
            "calls": critic_calls,
            "estimated_cost_usd": round(critic_calls * CRITIC_EST_USD_PER_CALL, 3),
            "budget_usd": critic_budget_usd,
            "verdicts": {
                "accept": sum(1 for row in written
                              if (row.get("critic") or {}).get("verdict") == "accept"),
                "revise": sum(1 for row in written
                              if (row.get("critic") or {}).get("verdict") == "revise"),
            },
            "reauthored": [row["figure_id"] for row in written
                           if (row.get("critic") or {}).get("reauthored")],
            "max_rounds": MAX_CRITIC_ROUNDS,
            # Major issues still standing after the last round: the operator
            # decides on these in the canary review; nothing loops further.
            # Open findings the operator has not yet decided on; a figure with
            # an operator_resolution (patch_figure_spec.py) is no longer open.
            "unresolved_major": [
                {"slide_id": row["slide_id"], "figure_id": row["figure_id"],
                 "issues": (row.get("critic") or {}).get("unresolved_major")}
                for row in written
                if (row.get("critic") or {}).get("unresolved_major")
                and not (row.get("critic") or {}).get("operator_resolution")
            ],
            "operator_resolved": [
                {"slide_id": row["slide_id"], **(row.get("critic") or {}).get("operator_resolution")}
                for row in written if (row.get("critic") or {}).get("operator_resolution")
            ],
            "errors": [
                {"slide_id": row["slide_id"], "error": (row.get("critic") or {}).get("error")}
                for row in written if (row.get("critic") or {}).get("error")
            ],
        },
        "blockers": blockers,
        "status": "pass" if written and not blockers else "blocked",
    }


def merge_targeted_receipt(
    deck_dir: Path, prior: dict[str, Any], update: dict[str, Any]
) -> dict[str, Any]:
    """Keep untouched slides active after a targeted authoring repair."""
    expected = [str(slide.get("slide_id") or "") for slide in _slide_rows(
        _load_json(deck_dir / "presentation_deck.json")
    )]
    prior_rows = prior.get("specs") if isinstance(prior.get("specs"), list) else []
    update_rows = update.get("specs") if isinstance(update.get("specs"), list) else []
    by_slide = {
        str(row.get("slide_id") or ""): row
        for row in prior_rows if isinstance(row, dict)
    }
    by_slide.update({
        str(row.get("slide_id") or ""): row
        for row in update_rows if isinstance(row, dict)
    })
    missing = [slide_id for slide_id in expected if slide_id not in by_slide]
    merged = dict(update)
    merged["slide_count"] = len(expected)
    merged["specs"] = [by_slide[slide_id] for slide_id in expected if slide_id in by_slide]
    merged["spec_count"] = len(merged["specs"])
    merged["targeted_update"] = [str(row.get("slide_id") or "") for row in update_rows]
    if missing:
        merged["blockers"] = [*update.get("blockers", []), {
            "code": "figure_spec_receipt_incomplete_after_targeted_update",
            "slide_ids": missing,
        }]
        merged["status"] = "blocked"
    return merged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-dir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="no provider call; write prompts and placeholders")
    parser.add_argument("--slide-limit", type=int, default=0)
    parser.add_argument(
        "--prefer", action="append", default=[], metavar="SLIDE_ID=ARCHETYPE",
        help="force one slide's archetype, e.g. --prefer sl_20=bar_compare. Repeatable. "
             "For the per-section quota, where the figure brief describes the mechanism "
             "and the section needs its measurement drawn.",
    )
    parser.add_argument(
        "--slide-ids", default="",
        help="comma-separated slide ids (or unique prefixes) to author; the rest are left alone. "
             "Re-authoring ten slides should not cost a run of forty-nine.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="leave slides that already have an authored spec untouched (placeholders are "
             "still refused and re-authored)",
    )
    parser.add_argument(
        "--spec-dir", type=Path,
        help="write specs here instead of <deck-dir>/figure_specs. Use it for --dry-run: "
             "a dry run writes placeholders, and into the real directory that clobbers "
             "authored work.",
    )
    parser.add_argument("--model", default=os.environ.get("FIGURE_SPEC_MODEL", ""),
                        help="override the governed authoring model (checked against the registry)")
    parser.add_argument("--provider", default=os.environ.get("FIGURE_SPEC_PROVIDER", ""),
                        help="authoring provider; default is the one the registry records for the model")
    parser.add_argument("--critic-model", default=os.environ.get("FIGURE_SPEC_CRITIC_MODEL", ""),
                        help="override the governed spec-critic model (checked against the registry)")
    parser.add_argument("--critic-provider", default=os.environ.get("FIGURE_SPEC_CRITIC_PROVIDER", ""),
                        help="critic provider; default is the one the registry records for the model")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--critic", action="store_true",
        help="accepted for compatibility: the adversarial spec critic is always on "
             f"(at most {MAX_CRITIC_ROUNDS} rounds per figure; a major issue buys a re-author)",
    )
    parser.add_argument(
        "--critic-budget-usd", type=float, default=6.0,
        help="cap on the critic's worst case (slides x rounds); a deck over it halts before spending",
    )
    parser.add_argument(
        "--workers", type=int, default=6,
        help=(
            "how many slides are authored at once (1 = one after another). Each "
            "slide's spec is independent, so the calls overlap; the receipt is "
            "assembled in slide order either way."
        ),
    )
    parser.add_argument("--no-critic-reauthor", action="store_true",
                        help="record the critique without acting on it")
    args = parser.parse_args(argv)

    try:
        with provider_receipt_context(
            workspace_root=args.workspace,
            stage_id="figure_spec_authoring",
            task_id=TASK_ID,
            caller=__file__,
        ):
            receipt = _generate_from_args(args)
    except CriticBudgetExceeded as error:
        print(f"halted before spending: {error}", file=sys.stderr)
        return 2
    return _finish(args, receipt)


def _generate_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return generate(
        deck_dir=args.deck_dir, workspace=args.workspace, dry_run=args.dry_run,
        slide_limit=args.slide_limit, model=args.model,
        provider=getattr(args, "provider", ""),
        critic_model=getattr(args, "critic_model", ""),
        critic_provider=getattr(args, "critic_provider", ""),
        skip_existing=args.skip_existing,
        slide_ids=tuple(part for part in args.slide_ids.split(",") if part.strip()),
        spec_dir=args.spec_dir,
        critic=True,
        critic_budget_usd=args.critic_budget_usd,
        workers=max(1, int(args.workers)),
        critic_reauthor=not args.no_critic_reauthor,
        preferred=dict(
            part.split("=", 1) for part in args.prefer if "=" in part
        ),
    )


def _finish(args: argparse.Namespace, receipt: dict[str, Any]) -> int:
    receipt_path = args.receipt or (args.deck_dir / RECEIPT_NAME)
    if args.slide_ids and args.spec_dir is None and not args.dry_run:
        # A targeted repair must not shrink the manifest that compile_figures
        # treats as the exact active deck. Preserve it on a failed attempt too.
        if receipt["status"] == "pass" and receipt_path.is_file():
            receipt = merge_targeted_receipt(
                args.deck_dir, _load_json(receipt_path), receipt
            )
        elif receipt["status"] == "pass":
            receipt["status"] = "blocked"
            receipt["blockers"] = [{
                "code": "figure_spec_targeted_update_requires_full_receipt",
            }]
        if receipt["status"] != "pass":
            receipt_path = args.deck_dir / "figure_spec_targeted_attempt_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(receipt, indent=2))
    else:
        print(f"{receipt['spec_count']}/{receipt['slide_count']} figure specs ({receipt['status']}); "
              f"{receipt['numeric_ledger_claims']} ledger claims; receipt {receipt_path}")
        for blocker in receipt["blockers"][:5]:
            print(f"  blocker {blocker['code']}: {blocker['slide_id']} {blocker['errors'][:1]}")
        critic_row = receipt.get("critic") or {}
        print(f"  adversarial critic: {critic_row.get('calls', 0)} calls, "
              f"{len(critic_row.get('reauthored') or [])} re-authored, "
              f"{len(critic_row.get('unresolved_major') or [])} with major issues left after "
              f"{critic_row.get('max_rounds')} rounds, {len(critic_row.get('errors') or [])} critic errors")
    return 0 if receipt["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
