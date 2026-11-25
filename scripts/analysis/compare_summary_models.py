#!/usr/bin/env python3
"""Send one Stage 07A topic/category chunk's exact prompt to several models.

Rebuilds the prompt ``exhaustive_summaries.py`` sent for one (topic, category)
pair -- same evidence collection, same chunking ceiling, same schema hint --
and asks each named catalog alias for the same synthesis pack. The pair's
stored output (the production model's reply) is the baseline. For every
model it records latency, token usage, cost, whether the reply parses into
the schema, how many of the cited unit_ids actually exist in the chunk, and
keeps every reply for side-by-side review. Models are named by catalog alias
(``model_for_alias``), never by literal. Nothing in ``analysis/`` is written.

    .venv/bin/python scripts/analysis/compare_summary_models.py \
        --workspace conferences/icml2026 --topic-id 09_data_memorization_privacy \
        --category papers --aliases kimi_openrouter,luna
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.analysis import exhaustive_summaries as es  # noqa: E402
from conference_pipeline.conference_identity import conference_name  # noqa: E402
from conference_pipeline.model_governance.registry import load_model_catalog, model_for_alias  # noqa: E402
from conference_pipeline.provider_receipts import (  # noqa: E402
    gemini_generate_content,
    openai_chat_completions_create,
    openrouter_chat_completions_create,
    provider_receipt_context,
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
SCHEMA_KEYS = (
    "summary",
    "key_points",
    "narrative",
    "notable_evidence",
    "tensions_or_contrasts",
    "open_questions",
    "analysis_feed",
)
_UNIT_ID_RE = re.compile(r"[A-Za-z0-9_:\-]{6,}")


def build_pair_prompt(workspace: Path, *, run_id: str, topic_id: str, category: str, chunk_index: int) -> tuple[str, es.CorpusChunk, es.TopicInfo]:
    topics = es._load_topic_info(workspace)
    topic = topics[topic_id]
    items = es.collect_topic_category_items(workspace, run_id=run_id, topic_id=topic_id, category=category)
    baseline = json.loads(es._summary_paths(workspace, topic_id, category)[0].read_text(encoding="utf-8"))
    # Chunk under the ceiling the stored pair was actually built with (the 07A
    # receipt records it); the code default has since moved below the price tier.
    receipt = json.loads((workspace / "storage" / "workflow-runs" / run_id / "07A-exhaustive-topic-summaries.json").read_text(encoding="utf-8"))
    max_input_tokens = int(receipt.get("max_input_tokens") or 0) or es.resolve_max_input_tokens(
        provider=str(baseline["provider"]), model=str(baseline["model"])).max_input_tokens
    chunks = es.chunk_evidence_items(topic_id=topic_id, category=category, items=items, max_input_tokens=max_input_tokens)
    if len(chunks) != int(baseline.get("chunk_count") or 0):
        raise SystemExit(f"rebuilt {len(chunks)} chunks but the stored pair has {baseline.get('chunk_count')}; chunking drifted")
    chunk = chunks[chunk_index - 1]
    stored = baseline["chunk_summaries"][chunk_index - 1]
    if int(stored.get("unit_count") or 0) != len(chunk.unit_ids):
        raise SystemExit(f"rebuilt chunk has {len(chunk.unit_ids)} units but stored chunk has {stored.get('unit_count')}")
    prompt = es.build_chunk_prompt(topic=topic, category=category, chunk=chunk, conference_label=conference_name(workspace, default="the conference"))
    return prompt, chunk, topic


def call_model(alias: str, *, prompt: str, max_output_tokens: int, reasoning_effort: str = "") -> dict[str, Any]:
    entry = load_model_catalog()[alias]
    provider = str(entry["provider"])
    model = model_for_alias(alias)
    started = time.monotonic()
    text, stop, usage, cost = "", None, {}, None
    messages = [{"role": "system", "content": "Return JSON only."}, {"role": "user", "content": prompt}]
    if provider == "openai":
        import openai

        result = openai_chat_completions_create(
            openai.OpenAI(timeout=1800), caller=__file__, model=model, messages=messages,
            response_format={"type": "json_object"}, **es._openai_options(model, max_output_tokens),
        )
    elif provider == "openrouter":
        import openai

        client = openai.OpenAI(base_url=OPENROUTER_BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"], timeout=1800)
        extra = {"extra_body": {"reasoning": {"effort": reasoning_effort}}} if reasoning_effort else {}
        result = openrouter_chat_completions_create(
            client, caller=__file__, model=model, messages=messages, max_tokens=max_output_tokens,
            response_format={"type": "json_object"}, **extra,
        )
    elif provider == "gemini":
        from google import genai

        result = gemini_generate_content(
            genai.Client(), caller=__file__, model=model, contents=prompt,
            config=es._provider_config(max_output_tokens),
        )
        text = str(getattr(result, "text", "") or "")
        candidates = getattr(result, "candidates", None) or []
        stop = str(getattr(candidates[0], "finish_reason", None)) if candidates else None
    else:
        raise SystemExit(f"alias {alias!r} is on provider {provider!r}, which this script does not call")
    if provider in {"openai", "openrouter"}:
        choice = (getattr(result, "choices", None) or [None])[0]
        message = getattr(choice, "message", None)
        text = str(getattr(message, "content", "") or "")
        stop = getattr(choice, "finish_reason", None)
        refusal = getattr(message, "refusal", None)
        if refusal:
            stop = f"{stop} (refusal: {str(refusal)[:200]})"
    receipt = getattr(result, "receipt", None)
    if receipt is not None:
        cost = getattr(receipt, "estimated_cost_usd", None)
        usage = {
            "input_tokens": getattr(receipt, "input_tokens", None),
            "output_tokens": getattr(receipt, "output_tokens", None),
            "receipt_path": str(getattr(receipt, "receipt_path", "") or ""),
        }
    return {
        "alias": alias, "provider": provider, "model": model, "stop_reason": stop,
        "seconds": round(time.monotonic() - started, 1), "chars": len(text), "text": text,
        "estimated_cost_usd": cost, **usage,
    }


def score(payload: dict[str, Any], *, chunk_unit_ids: set[str]) -> dict[str, Any]:
    parsed = "raw_text" not in payload and bool(payload)
    missing = [k for k in SCHEMA_KEYS if k not in payload]
    cited: list[str] = []
    for ev in payload.get("notable_evidence") or []:
        if isinstance(ev, dict):
            cited.extend(str(u) for u in (ev.get("unit_ids") or []))
    # unit_ids are also sprinkled through prose (e.g. "(E5SVowO13b)"): count every chunk unit mentioned anywhere.
    blob = json.dumps(payload, ensure_ascii=True)
    mentioned = {u for u in chunk_unit_ids if u in blob}
    short_ids = {u.rsplit(":", 1)[-1] for u in chunk_unit_ids}
    short_mentioned = {s for s in short_ids if s in blob}
    valid_cited = [u for u in cited if u in chunk_unit_ids or u.rsplit(":", 1)[-1] in short_ids]
    return {
        "parsed": parsed,
        "missing_keys": missing,
        "key_points": len(payload.get("key_points") or []),
        "notable_evidence": len(payload.get("notable_evidence") or []),
        "summary_chars": len(str(payload.get("summary") or "")),
        "narrative_chars": len(str(payload.get("narrative") or "")),
        "cited_unit_ids": len(cited),
        "cited_unit_ids_valid": len(valid_cited),
        "cited_unit_ids_invalid": sorted(set(cited) - set(valid_cited))[:20],
        "distinct_chunk_units_mentioned": len(mentioned | {u for u in chunk_unit_ids if u.rsplit(":", 1)[-1] in short_mentioned}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--run-id", default=es.DEFAULT_RUN_ID)
    parser.add_argument("--topic-id", required=True)
    parser.add_argument("--category", default="papers", choices=es.VALID_CATEGORIES)
    parser.add_argument("--chunk-index", type=int, default=1)
    parser.add_argument("--aliases", required=True, help="comma-separated catalog aliases")
    parser.add_argument("--max-output-tokens", type=int, default=es.DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--reasoning-effort", default="", help="OpenRouter reasoning effort (low|medium|high)")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    prompt, chunk, topic = build_pair_prompt(workspace, run_id=args.run_id, topic_id=args.topic_id, category=args.category, chunk_index=args.chunk_index)
    baseline_path = es._summary_paths(workspace, args.topic_id, args.category)[0]
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    stored_chunk = baseline["chunk_summaries"][args.chunk_index - 1]
    chunk_units = set(chunk.unit_ids)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = args.out_dir or (workspace / "storage" / "workflow-runs" / args.run_id / f"summary-model-comparison-{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    (out_dir / "baseline.json").write_text(json.dumps(stored_chunk, indent=2, sort_keys=True), encoding="utf-8")
    results: list[dict[str, Any]] = [{
        "alias": "(baseline)", "provider": baseline["provider"], "model": baseline["model"],
        "estimated_cost_usd": None, "seconds": None, "stop_reason": "stored",
        **score(stored_chunk, chunk_unit_ids=chunk_units),
    }]
    print(f"prompt: {chunk.input_tokens:,} tokens, {len(chunk.unit_ids)} units; baseline {baseline['model']}", flush=True)

    with provider_receipt_context(workspace_root=workspace, run_id=args.run_id, stage_id="summary-model-comparison", task_id=es.OWNER_ID, caller=__file__):
        for alias in [a.strip() for a in args.aliases.split(",") if a.strip()]:
            print(f"-> {alias} ...", flush=True)
            try:
                reply = call_model(alias, prompt=prompt, max_output_tokens=args.max_output_tokens, reasoning_effort=args.reasoning_effort)
            except Exception as exc:  # keep going; the error is the result
                reply = {"alias": alias, "model": model_for_alias(alias), "error": f"{type(exc).__name__}: {str(exc)[:500]}", "text": ""}
            text = reply.pop("text", "")
            (out_dir / f"{alias}.txt").write_text(text, encoding="utf-8")
            payload = es.parse_json_object(text) if text else {}
            if "raw_text" not in payload and payload:
                (out_dir / f"{alias}.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            row = {**reply, **score(payload, chunk_unit_ids=chunk_units)}
            results.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "cited_unit_ids_invalid"}, default=str), flush=True)

    summary = {
        "topic_id": args.topic_id, "topic_title": topic.title, "category": args.category,
        "chunk_index": args.chunk_index, "chunk_count": chunk.chunk_count,
        "input_tokens": chunk.input_tokens, "unit_count": len(chunk.unit_ids),
        "baseline_path": str(baseline_path), "results": results,
    }
    (out_dir / "comparison.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
