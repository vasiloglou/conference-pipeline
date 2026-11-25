#!/usr/bin/env python3
"""Summarize each ICLR topic's narration into executive-readout digests with Opus.

For every conference topic this reads the topic's presenter narration and blueprint
thesis and asks Claude Opus to produce a source-grounded executive digest at three
levels of zoom:

* ``central_takeaway`` + ``one_liner`` — for the central-message and 12-topic index slides
* ``short`` — an evidence slide (named sources + a real diagram spec) and a
  big-picture slide (leadership implication + a GenAI hero-image concept)
* ``long`` — three deeper sub-theme slides

A final global pass synthesizes the conference-wide central message, open questions,
and predictions from the twelve topic digests.

Output: ``decks/executive_topic_digests.json`` (+ a receipt). This is the content
spine the executive-summary decks are rendered from — it replaces the hand-authored
``executive_summary_deck_first_plan.json`` lesson list so the deck captures every
topic's essence, grounded in named conference sources.

The deck render is intentionally NOT performed here; this stage only produces the
machine-checkable content contract.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from conference_pipeline.conference_identity import conference_name  # noqa: E402
from conference_pipeline.model_governance.llm_client import (  # noqa: E402
    complete,
    credential_names,
    has_credentials,
    resolve_selection,
)

TASK_ID = "presentation.executive_topic_digest"
DEFAULT_MODEL = resolve_selection(TASK_ID)[1]
DEFAULT_OUTPUT = Path("decks/executive_topic_digests.json")
NARRATION_CHAR_CAP = 26000

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_topics(workspace: Path) -> list[dict[str, str]]:
    manifest = json.loads((workspace / "analysis" / "analysis_manifest.json").read_text())
    topics: list[dict[str, str]] = []
    for row in manifest.get("topics", []):
        topic_id = str(row.get("id") or row.get("slug"))
        topics.append(
            {
                "topic_id": topic_id,
                "title": str(row.get("title") or topic_id),
                "number": str(row.get("number") or ""),
                "deck_dir": str(row.get("decks_dir") or f"decks/{topic_id}"),
            }
        )
    return topics


def read_topic_inputs(workspace: Path, topic: dict[str, str]) -> dict[str, str]:
    deck = workspace / topic["deck_dir"]
    narration = ""
    script = deck / "6_presenter_script.md"
    if script.is_file():
        narration = script.read_text(encoding="utf-8")[:NARRATION_CHAR_CAP]
    thesis = conclusion = promise = ""
    blueprint = deck / "presentation_blueprint.json"
    if blueprint.is_file():
        bp = json.loads(blueprint.read_text(encoding="utf-8"))
        thesis = str(bp.get("deck_thesis", ""))
        conclusion = str(bp.get("conclusion_synthesis", ""))
        promise = str(bp.get("audience_promise", ""))
    return {"narration": narration, "thesis": thesis, "conclusion": conclusion, "promise": promise}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# The diagram vocabulary the figure compiler can draw from flat labels, and
# the label conventions each kind needs. Stated once and reused by every
# diagram-authoring prompt so the two never drift. A kind outside this list
# renders as a plain list; a kind whose convention is not followed falls back
# the same way (see conference_pipeline.figures.from_diagram) -- which is how
# 11 executive figures shipped as numbered lists when their captions promised
# forks, loops and two-axis maps.
DIAGRAM_KINDS = (
    "one of: pipeline | mechanism | architecture | comparison | branch | loop | quadrant | "
    "ladder | matrix. Use 'comparison' for A-versus-B, 'branch' for one input that forks "
    "into alternatives (routing, two answer paths), 'loop' for a cycle that returns to its "
    "start, 'quadrant' for a two-axis map, 'matrix' for one hub with satellites."
)
DIAGRAM_NODE_RULES = (
    "3-5 short labels (<= 8 words each) for a REAL diagram, not bullet prose. CONVENTIONS: "
    "'comparison' -> every label prefixed by its side, exactly two sides, e.g. "
    "'Scale-first: one universal embedding' / 'Structure-first: routed experts'. "
    "'branch' -> first label is the input, every other label prefixed by its branch, e.g. "
    "'Input scored by a router', 'Easy: exit cheaply', 'Hard: escalate'. "
    "'loop' -> labels in cycle order; the last returns to the first. "
    "'quadrant' -> 'X: <low> | <high>', 'Y: <low> | <high>', then labels placed as "
    "'top-left: ...', 'bottom-right: ...'. 'matrix' -> add \"hub\": a <= 4 word centre label, "
    "and \"spoke_direction\": \"to_hub\" when the labels feed or make up the hub (criteria "
    "behind a decision), \"from_hub\" when they respond to or branch from it (fixes for a "
    "problem, extensions of a baseline). 'pipeline' only for steps that really run in order; "
    "alternatives or independent choices are a 'matrix'. "
    "The caption must count what the nodes show: never 'four fixes' over five labels."
)


def build_topic_schema(conference_label: str, topic_count: int) -> dict[str, Any]:
    return {
    "central_takeaway": f"ONE sentence: the single most important thing {conference_label} showed in this topic.",
    "one_liner": f"<= 14 words, punchy, for a {topic_count}-topic index slide. No trailing period required.",
    "short": {
        "evidence_slide": {
            "headline": "<= 8 words: the finding.",
            "synthesis": "2-3 sentences, concrete, naming mechanisms/results. No ellipses.",
            "named_sources": "list of 2-4 {title, why} citing REAL papers/speakers from the narration. Never invent.",
            "diagram": {
                "kind": DIAGRAM_KINDS,
                "caption": "one line describing what the diagram depicts",
                "nodes": DIAGRAM_NODE_RULES,
            },
        },
        "bigpicture_slide": {
            "headline": "<= 8 words: the so-what.",
            "implication": "1-2 sentences: a concrete decision/implication for technical leaders.",
            "image_concept": "a source-grounded concept for a hero image with NO text, evoking the mechanism.",
        },
    },
    "long": {
        # The plan builder renders every long slide as a tech_diagram, so each
        # one needs a diagram spec; without it the figure card is a bare arrow.
        "slides": "EXACTLY 3 items of {headline (<=8 words), body (3-4 sentences, concrete), "
        "named_sources (list of {title, why}), diagram ({kind, caption, nodes} exactly as in "
        "short.evidence_slide.diagram, depicting THIS sub-theme's mechanism)}: deeper "
        "sub-themes than the short cut.",
    },
}

def build_topic_system(conference_label: str) -> str:
    return (
    f"You are an expert analyst writing an executive readout of {conference_label} for technical leaders "
    "(research directors, eng VPs, founders). Ground every claim in the provided topic narration. "
    "Cite ONLY papers, works, and speakers that actually appear in the narration; never invent "
    "titles or names. Be specific and concrete: name mechanisms, results, and people. Avoid generic "
    "best-practice filler and avoid restating the topic title as the insight. Text must read in full "
    "with no ellipsis truncation. Return ONLY a single JSON object matching the schema, no prose."
)


def build_topic_user(
    topic: dict[str, str], inputs: dict[str, str], *, conference_label: str, topic_count: int
) -> str:
    return (
        f"TOPIC: {topic['title']}\n\n"
        f"DECK THESIS:\n{inputs['thesis']}\n\n"
        f"AUDIENCE PROMISE:\n{inputs['promise']}\n\n"
        f"CONCLUSION SYNTHESIS:\n{inputs['conclusion']}\n\n"
        f"FULL TOPIC NARRATION (named-source dense):\n{inputs['narration']}\n\n"
        f"Produce the executive digest as JSON with EXACTLY these fields:\n"
        f"{json.dumps(build_topic_schema(conference_label, topic_count), indent=2)}"
    )


def build_global_system(conference_label: str, topic_count: int) -> str:
    return (
    f"You are the editor-in-chief assembling the {conference_label} executive readout from {topic_count} topic digests. "
    "Synthesize ACROSS topics; do not just concatenate. Be specific and name the throughlines that "
    "recur across topics. Return ONLY a JSON object, no prose."
)


def build_global_schema(conference_label: str, topic_count: int) -> dict[str, Any]:
    return {
    "central_message": {
        "headline": f"<= 10 words: the single thesis of {conference_label} across all {topic_count} topics.",
        "subtitle": "1-2 sentences expanding the thesis, referencing the cross-topic throughlines.",
    },
    "open_questions": "list of EXACTLY 3 {question, why_it_matters}: the biggest unresolved questions the conference surfaced.",
    "predictions": "list of EXACTLY 3 {prediction, basis}: concrete near-term predictions grounded in the conference evidence.",
    # The opening index slide reads as a guided itinerary, not a flat directory,
    # so the topics are grouped into narrative arcs. These MUST come from this
    # conference's own topics: an earlier implementation carried a hardcoded
    # table of one conference's topic slugs, which silently produced no arcs
    # for every other conference and collapsed the index slide into an
    # unreadable 12+26-item bookend. See ARC_CONTRACT below.
    "arcs": (
        f"list of 3-4 {{name, topic_ids}} grouping ALL {topic_count} topics into narrative arcs. "
        "'name' is <= 6 words naming the throughline (e.g. 'Agents you can trust to act'). "
        "'topic_ids' is a list of topic_id strings copied VERBATIM from the input. "
        "Every topic_id must appear in EXACTLY ONE arc; every arc must be non-empty; "
        "invent no topic_ids. Order the arcs so the index tells a story."
    ),
}


# The deterministic contract the model's `arcs` must satisfy. Stated once here
# so the generator, the plan builder, and the tests all cite the same rule.
ARC_CONTRACT = (
    "arcs must be 3-4 non-empty groups partitioning every topic_id exactly once"
)
ARC_MIN = 3
ARC_MAX = 4


def validate_arcs(arcs: Any, topic_ids: Sequence[str]) -> list[dict[str, Any]]:
    """Return normalized arcs, or raise if they do not partition ``topic_ids``.

    Fails closed on purpose. A partial or empty arc grouping is exactly the
    state that produced the CVPR 2026 executive slide-2 defect: the renderer
    keys its two-column itinerary layout off a non-empty ``arc`` on every row,
    and silently falls back to the 5-item topic-deck bookend when any row
    lacks one. There is no sensible partial arc grouping, so there is no
    partial success here.
    """

    expected = list(dict.fromkeys(str(t) for t in topic_ids))
    if not isinstance(arcs, list) or not arcs:
        raise ValueError(f"global synthesis returned no 'arcs'; {ARC_CONTRACT}")
    if not ARC_MIN <= len(arcs) <= ARC_MAX:
        raise ValueError(
            f"global synthesis returned {len(arcs)} arcs; {ARC_CONTRACT}"
        )
    normalized: list[dict[str, Any]] = []
    assigned: list[str] = []
    for entry in arcs:
        if not isinstance(entry, dict):
            raise ValueError(f"arc entry is not an object: {entry!r}; {ARC_CONTRACT}")
        name = " ".join(str(entry.get("name") or "").split())
        raw_ids = entry.get("topic_ids")
        if not name:
            raise ValueError(f"arc entry has no name: {entry!r}; {ARC_CONTRACT}")
        if not isinstance(raw_ids, list) or not raw_ids:
            raise ValueError(f"arc {name!r} has no topic_ids; {ARC_CONTRACT}")
        ids = [str(value).strip() for value in raw_ids]
        unknown = [value for value in ids if value not in expected]
        if unknown:
            raise ValueError(
                f"arc {name!r} cites topic_ids that are not this conference's: "
                f"{unknown}; {ARC_CONTRACT}"
            )
        assigned.extend(ids)
        normalized.append({"name": name, "topic_ids": ids})
    duplicated = sorted({value for value in assigned if assigned.count(value) > 1})
    if duplicated:
        raise ValueError(f"topic_ids appear in more than one arc: {duplicated}; {ARC_CONTRACT}")
    missing = [value for value in expected if value not in assigned]
    if missing:
        raise ValueError(f"topic_ids missing from every arc: {missing}; {ARC_CONTRACT}")
    return normalized


def build_global_user(
    digests: list[dict[str, Any]], *, conference_label: str, topic_count: int
) -> str:
    rows = []
    for d in digests:
        short = d.get("short", {})
        rows.append(
            {
                # topic_id is what `arcs` must cite verbatim, so the model has
                # to see it. Without it the model can only guess at slugs.
                "topic_id": d.get("topic_id"),
                "topic": d.get("title"),
                "central_takeaway": d.get("central_takeaway"),
                "one_liner": d.get("one_liner"),
                "evidence_headline": short.get("evidence_slide", {}).get("headline"),
                "implication": short.get("bigpicture_slide", {}).get("implication"),
            }
        )
    return (
        f"ALL {topic_count} TOPIC DIGESTS:\n"
        + json.dumps(rows, indent=2, ensure_ascii=False)
        + "\n\nProduce the conference-wide synthesis as JSON with EXACTLY these fields:\n"
        + json.dumps(build_global_schema(conference_label, topic_count), indent=2)
    )


# ---------------------------------------------------------------------------
# Opus calls
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object found in model response")
    return json.loads(match.group(0))


def call_model(
    *, system: str, user: str, model: str, max_tokens: int, caller: str
) -> tuple[dict[str, Any], dict[str, int], str | None]:
    """One digest call, repaired in conversation, through whichever provider the task allows ``model`` on."""

    # A malformed reply is repaired in conversation: the model sees its own
    # reply and the parser's error. Re-asking the same prompt with "return
    # only JSON" reproduced the same unescaped quote twice (Opus 5.5,
    # 2026-09-25); a reply cut off at max_tokens gets a larger budget.
    messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
    usage_d = {"input_tokens": 0, "output_tokens": 0}
    last_err: Exception | None = None
    for attempt in range(3):
        result = complete(
            TASK_ID,
            system=system,
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            caller=caller,
        )
        text = result.text
        usage_d["input_tokens"] += result.input_tokens
        usage_d["output_tokens"] += result.output_tokens
        receipt_path = getattr(result.receipt, "receipt_path", None)
        try:
            return _extract_json(text), usage_d, (str(receipt_path) if receipt_path else None)
        except (ValueError, json.JSONDecodeError) as exc:  # noqa: PERF203
            last_err = exc
            if result.stop_reason == "max_tokens":
                max_tokens *= 2
                messages = [{"role": "user", "content": user}]
                continue
            messages = [
                {"role": "user", "content": user},
                {"role": "assistant", "content": text},
                {
                    "role": "user",
                    "content": (
                        f"That reply is not valid JSON ({exc}). Return the complete corrected "
                        "JSON object only -- escape any double quote inside a string value."
                    ),
                },
            ]
    raise RuntimeError(f"{model} did not return valid JSON after retries: {last_err}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def require_credentials(model: str) -> None:
    provider, _ = resolve_selection(TASK_ID, None, model)
    if not has_credentials(provider):
        raise RuntimeError(
            f"{' or '.join(credential_names(provider) or ())} is required for {provider} "
            "(source the repo .env first)"
        )


def run(workspace: Path, *, model: str, only: list[str] | None, limit: int | None) -> dict[str, Any]:
    topics = load_topics(workspace)
    conference_label = conference_name(workspace, default="the conference")
    topic_count = len(topics)
    if only:
        topics = [t for t in topics if t["topic_id"] in only or t["title"] in only]
    if limit:
        topics = topics[:limit]
    require_credentials(model)

    digests: list[dict[str, Any]] = []
    usages: list[dict[str, Any]] = []
    for topic in topics:
        inputs = read_topic_inputs(workspace, topic)
        if not inputs["narration"]:
            print(f"  ! {topic['topic_id']}: no narration found, skipping", file=sys.stderr)
            continue
        digest, usage, receipt = call_model(
            system=build_topic_system(conference_label),
            user=build_topic_user(
                topic, inputs, conference_label=conference_label, topic_count=topic_count
            ),
            model=model,
            max_tokens=3200,
            caller="generate_executive_topic_digests:topic",
        )
        digest["topic_id"] = topic["topic_id"]
        digest["title"] = topic["title"]
        digest["number"] = topic["number"]
        digests.append(digest)
        usages.append({"topic_id": topic["topic_id"], "stage": "topic", "receipt": receipt, **usage})
        print(
            f"  + {topic['topic_id']}: {digest.get('one_liner','')!r} "
            f"(in={usage['input_tokens']} out={usage['output_tokens']})",
            file=sys.stderr,
        )

    global_synth: dict[str, Any] = {}
    if len(digests) >= 2:
        global_synth, usage, receipt = call_model(
            system=build_global_system(conference_label, topic_count),
            user=build_global_user(
                digests, conference_label=conference_label, topic_count=topic_count
            ),
            model=model,
            max_tokens=2000,
            caller="generate_executive_topic_digests:global",
        )
        usages.append({"topic_id": "__global__", "stage": "global", "receipt": receipt, **usage})

    # Validate before writing: a digest file carrying unusable arcs would be
    # discovered three scripts later, at render time, as a silently wrong
    # layout rather than an error.
    arcs = validate_arcs(global_synth.get("arcs"), [d["topic_id"] for d in digests])
    arc_by_topic = {
        topic_id: entry["name"] for entry in arcs for topic_id in entry["topic_ids"]
    }

    return {
        "schema_version": 1,
        "generated_at": utc_now(),
        "model": model,
        "central_message": global_synth.get("central_message", {}),
        "open_questions": global_synth.get("open_questions", []),
        "predictions": global_synth.get("predictions", []),
        "arcs": arcs,
        "topic_index": [
            {
                "topic_id": d["topic_id"],
                "title": d["title"],
                "one_liner": d.get("one_liner", ""),
                "arc": arc_by_topic[d["topic_id"]],
            }
            for d in digests
        ],
        "topics": digests,
        "usage": usages,
        "usage_totals": {
            "input_tokens": sum(u["input_tokens"] for u in usages),
            "output_tokens": sum(u["output_tokens"] for u in usages),
            "topic_count": len(digests),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, required=True,
                   help="Conference workspace dir (e.g. conferences/<slug>).")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="relative to --workspace")
    p.add_argument("--topic", action="append", dest="topics", help="restrict to topic id(s); repeatable")
    p.add_argument("--limit", type=int, default=None, help="only the first N topics (smoke runs)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--arcs-only",
        action="store_true",
        help=(
            "Backfill narrative arcs into an EXISTING digest file without "
            "regenerating any topic digest. Runs the cross-topic synthesis call "
            "and merges only its 'arcs' -- every topic digest, the central "
            "message, open questions and predictions are preserved byte for "
            "byte, so deck narration (and its TTS audio) does not change. Use "
            "this to upgrade a conference whose digests predate arcs."
        ),
    )
    return p.parse_args(argv)


def backfill_arcs(workspace: Path, digest_path: Path, *, model: str) -> dict[str, Any]:
    """Add arcs to an existing digest file, changing nothing else."""

    payload = json.loads(digest_path.read_text(encoding="utf-8"))
    digests = payload.get("topics") or []
    if len(digests) < 2:
        raise ValueError(f"{digest_path} has fewer than 2 topic digests; nothing to group")
    conference_label = conference_name(workspace, default="the conference")
    require_credentials(model)
    global_synth, usage, receipt = call_model(
        system=build_global_system(conference_label, len(digests)),
        user=build_global_user(
            digests, conference_label=conference_label, topic_count=len(digests)
        ),
        model=model,
        max_tokens=2000,
        caller="generate_executive_topic_digests:arcs_backfill",
    )
    arcs = validate_arcs(global_synth.get("arcs"), [d["topic_id"] for d in digests])
    arc_by_topic = {tid: entry["name"] for entry in arcs for tid in entry["topic_ids"]}

    payload["arcs"] = arcs
    payload["topic_index"] = [
        {**row, "arc": arc_by_topic[row["topic_id"]]}
        for row in payload.get("topic_index", [])
        if row.get("topic_id")
    ]
    usages = payload.setdefault("usage", [])
    if isinstance(usages, list):
        usages.append(
            {"topic_id": "__arcs__", "stage": "arcs_backfill", "receipt": receipt, **usage}
        )
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workspace = args.workspace.expanduser().resolve()
    # D-QUICK-260715-rz1: establish a workspace-rooted receipt context so
    # standalone invocations don't fall back to Path.cwd() (repo root).
    os.environ.setdefault("PROVIDER_RECEIPT_WORKSPACE", str(workspace))
    os.environ.setdefault("PROVIDER_RECEIPT_STAGE_ID", "generate_executive_topic_digests")
    output = (workspace / args.output).resolve()
    if args.dry_run:
        topics = load_topics(workspace)
        print(json.dumps({"workspace": str(workspace), "model": args.model,
                          "output": str(output), "topic_count": len(topics),
                          "topics": [t["topic_id"] for t in topics], "status": "planned"}, indent=2))
        return 0
    if args.arcs_only:
        result = backfill_arcs(workspace, output, model=args.model)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(
            f"Backfilled {len(result['arcs'])} arcs into {output} "
            f"({len(result['topic_index'])} topics tagged); no topic digest regenerated",
            file=sys.stderr,
        )
        print(str(output))
        return 0
    result = run(workspace, model=args.model, only=args.topics, limit=args.limit)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    totals = result["usage_totals"]
    print(
        f"Wrote {output} — {totals['topic_count']} topics, "
        f"in={totals['input_tokens']} out={totals['output_tokens']} tokens, model={args.model}",
        file=sys.stderr,
    )
    print(str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
