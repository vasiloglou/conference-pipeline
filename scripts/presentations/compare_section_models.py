#!/usr/bin/env python3
"""Send one editorial-blueprint section's exact prompt to several models.

Built for a section the blueprint model declined (ICML 2026 deck 01's
molecule section, 2026-10-01): it rebuilds the section prompt the generator
sends, from the deck's saved editorial board, and asks each named catalog
alias for the same slides. For every model it records whether it declined,
how much it wrote, whether the reply parses into slides, and whether those
slides cite only the section's own sources, and it keeps every reply so the
slides can be compared side by side. Models are named by catalog alias
(``model_for_alias``), never by literal. Nothing in the deck is written.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.conference_identity import conference_name  # noqa: E402
from conference_pipeline.model_governance.registry import load_model_catalog, model_for_alias  # noqa: E402
from conference_pipeline.provider_receipts import (  # noqa: E402
    anthropic_messages_create,
    gemini_generate_content,
    openai_chat_completions_create,
    openrouter_chat_completions_create,
    provider_receipt_context,
)

TASK_ID = "presentation.blueprint_generation"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MAX_TOKENS = 14000


def _load_generator() -> Any:
    path = PROJECT_ROOT / "scripts" / "presentations" / "generate_editorial_blueprint.py"
    spec = importlib.util.spec_from_file_location("editorial_blueprint_for_comparison", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def section_prompt(module: Any, *, workspace: Path, deck_dir: Path, analysis_md: Path, section_id: str) -> tuple[str, str, set[str]]:
    """The generator's own prompt for one section, and the section's source ids."""

    saved = json.loads((deck_dir / "editorial_board_artifacts.json").read_text(encoding="utf-8"))
    board = saved["board"]
    ledger = json.loads((deck_dir / "speaker_author_work_ledger.json").read_text(encoding="utf-8"))
    ledger_index = {str(e.get("source_id")): e for e in ledger.get("entries") or [] if isinstance(e, dict)}
    catalog = module.build_source_catalog(ledger)
    catalog_by_id = {c["source_id"]: c for c in catalog}
    selected = {str(s.get("source_id")) for s in board.get("selected_sources") or []} & set(ledger_index)
    clusters = {str(c.get("cluster_id")): c for c in board.get("clusters") or []}
    sections = [s for s in board.get("sections") or [] if isinstance(s, dict)]
    section = next((s for s in sections if str(s.get("section_id")) == section_id), None)
    if section is None:
        raise SystemExit(f"section {section_id!r} not in the saved board: {[s.get('section_id') for s in sections]}")
    source_ids = [
        sid for cid in section.get("cluster_ids") or []
        for sid in (clusters.get(str(cid)) or {}).get("source_ids") or [] if sid in selected
    ]
    section_sources = [catalog_by_id[s] for s in dict.fromkeys(source_ids) if s in catalog_by_id]
    prior_titles: list[str] = []
    for saved_section, slides in saved.get("section_slides") or []:
        if str(saved_section.get("section_id")) == section_id:
            break
        prior_titles.extend(str(s.get("title") or "") for s in slides)

    captured: list[str] = []
    module._call = lambda client, prompt, *, max_tokens=8000: captured.append(prompt) or []
    module.author_section_slides(
        None,
        topic_title=str(
            json.loads((deck_dir / "presentation_blueprint.json").read_text(encoding="utf-8")).get("retrieval_title")
            or deck_dir.name.replace("_", " ")
        ),
        section=section,
        section_sources=section_sources,
        evidence=module.build_evidence_digest(analysis_md, None),
        prior_titles=prior_titles[-12:],
        presentation_frame=None,
        conference_label=conference_name(workspace, default="the conference"),
    )
    return captured[0], str(section.get("title") or section_id), {c["source_id"] for c in section_sources}


def _system_prompt(module: Any) -> str:
    return (
        "You are the editorial board for a technical conference talk. "
        "You ground every claim in the provided named sources. "
        "Respond with pure JSON only — no prose, no markdown fences."
    )


def call_model(alias: str, *, system: str, prompt: str, reasoning_effort: str = "") -> dict[str, Any]:
    entry = load_model_catalog()[alias]
    provider = str(entry["provider"])
    model = model_for_alias(alias)
    started = time.monotonic()
    text, stop = "", None
    if provider == "anthropic":
        import anthropic

        result = anthropic_messages_create(
            anthropic.Anthropic(), caller=__file__, model=model, max_tokens=MAX_TOKENS,
            system=system, messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(getattr(b, "text", "") for b in getattr(result, "content", []) or [])
        stop = getattr(result, "stop_reason", None)
    elif provider in {"openai", "openrouter"}:
        import openai

        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        if provider == "openai":
            result = openai_chat_completions_create(
                openai.OpenAI(), caller=__file__, model=model, messages=messages,
                max_completion_tokens=MAX_TOKENS,
            )
        else:
            client = openai.OpenAI(base_url=OPENROUTER_BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"])
            # A reasoning model can spend the whole token allowance thinking and
            # return no text (finish_reason "length"); a lower effort leaves room.
            extra = {"extra_body": {"reasoning": {"effort": reasoning_effort}}} if reasoning_effort else {}
            result = openrouter_chat_completions_create(
                client, caller=__file__, model=model, messages=messages, max_tokens=MAX_TOKENS, **extra,
            )
        choice = (getattr(result, "choices", None) or [None])[0]
        message = getattr(choice, "message", None)
        text = str(getattr(message, "content", "") or "")
        stop = getattr(choice, "finish_reason", None)
        refusal = getattr(message, "refusal", None)
        if refusal:
            stop = f"{stop} (refusal: {str(refusal)[:200]})"
    elif provider == "gemini":
        from google import genai
        from google.genai import types

        result = gemini_generate_content(
            genai.Client(), caller=__file__, model=model, contents=prompt,
            config=types.GenerateContentConfig(system_instruction=system, max_output_tokens=MAX_TOKENS),
        )
        text = str(getattr(result, "text", "") or "")
        candidates = getattr(result, "candidates", None) or []
        stop = getattr(candidates[0], "finish_reason", None) if candidates else None
        feedback = getattr(result, "prompt_feedback", None)
        if feedback is not None and getattr(feedback, "block_reason", None):
            stop = f"blocked: {feedback.block_reason}"
    else:
        raise SystemExit(f"alias {alias!r}: provider {provider!r} is not supported here")
    return {"alias": alias, "provider": provider, "model": model, "text": text,
            "stop_reason": str(stop) if stop is not None else None,
            "seconds": round(time.monotonic() - started, 1)}


def assess(module: Any, reply: dict[str, Any], *, source_ids: set[str], budget: int) -> dict[str, Any]:
    text = reply["text"]
    row = {k: reply[k] for k in ("alias", "provider", "model", "stop_reason", "seconds")}
    row["chars"] = len(text)
    try:
        payload = module._extract_json(text) if text.strip() else None
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        payload = payload.get("slides") if isinstance(payload.get("slides"), list) else [payload]
    slides = [s for s in payload or [] if isinstance(s, dict) and s.get("title")]
    cited = [str((m or {}).get("source_id") or "") for s in slides for m in s.get("mentions") or []]
    row.update(
        parsed=payload is not None,
        slides=len(slides),
        budget=budget,
        mentions=len(cited),
        foreign_mentions=sorted({c for c in cited if c not in source_ids}),
        sources_used=len({c for c in cited if c in source_ids}),
        sources_available=len(source_ids),
        titles=[str(s.get("title")) for s in slides],
        declined=not slides,
    )
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--deck-dir", type=Path, required=True)
    parser.add_argument("--analysis-md", type=Path, required=True)
    parser.add_argument("--section-id", required=True)
    parser.add_argument("--aliases", required=True, help="comma-separated catalog aliases")
    parser.add_argument("--repeats", type=int, default=1, help="calls per model (refusals can be stochastic)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reasoning-effort", default="", help="OpenRouter reasoning effort (low|medium|high)")
    args = parser.parse_args(argv)

    module = _load_generator()
    prompt, title, source_ids = section_prompt(
        module, workspace=args.workspace, deck_dir=args.deck_dir,
        analysis_md=args.analysis_md, section_id=args.section_id,
    )
    budget = 0
    for line in prompt.splitlines():
        if line.startswith("Author exactly "):
            budget = int(line.split()[2])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    rows: list[dict[str, Any]] = []
    with provider_receipt_context(
        workspace_root=args.workspace, stage_id="editorial_blueprint_model_comparison",
        task_id=TASK_ID, caller=__file__,
    ):
        for alias in [a.strip() for a in args.aliases.split(",") if a.strip()]:
            for attempt in range(1, max(1, args.repeats) + 1):
                try:
                    reply = call_model(alias, system=_system_prompt(module), prompt=prompt,
                                       reasoning_effort=args.reasoning_effort)
                except Exception as error:  # noqa: BLE001 - one model failing must not stop the comparison
                    reply = {"alias": alias, "provider": "?", "model": "?", "text": "",
                             "stop_reason": f"error: {type(error).__name__}: {str(error)[:300]}", "seconds": 0}
                (args.output_dir / f"{alias}.{attempt}.txt").write_text(reply["text"], encoding="utf-8")
                row = assess(module, reply, source_ids=source_ids, budget=budget)
                row["attempt"] = attempt
                rows.append(row)
                print(json.dumps({k: row[k] for k in ("alias", "attempt", "stop_reason", "chars", "slides", "foreign_mentions")}))
    summary = {"section_id": args.section_id, "section_title": title, "slide_budget": budget,
               "source_ids": sorted(source_ids), "results": rows}
    (args.output_dir / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
