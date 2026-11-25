#!/usr/bin/env python3
"""Bounded, offline inference comparison using a frozen governed model manifest.

The controller resolves model selections in this repository. Workers can run on
another Mac without copying the repository or credentials. No production writes.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def fingerprint(text):
    return hashlib.sha256(text.encode()).hexdigest()


def versions():
    found = {}
    for name in ("mlx", "mlx-lm", "vllm", "vllm-metal", "vllm-mlx", "transformers"):
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return found


def memory():
    return {
        "swap": subprocess.check_output(["sysctl", "vm.swapusage"], text=True).strip(),
        "load_average": subprocess.check_output(["sysctl", "vm.loadavg"], text=True).strip(),
        "vm_stat": subprocess.check_output(["vm_stat"], text=True),
    }


def plan(args):
    from huggingface_hub import snapshot_download
    from conference_pipeline.model_governance.registry import (
        load_model_catalog, model_for_alias, validate_selection,
    )
    task = "analysis.local_summary_benchmark"
    models = []
    for alias in args.aliases.split(","):
        model = model_for_alias(alias)
        validate_selection(task, provider="local", model=model)
        path = snapshot_download(model, revision=load_model_catalog()[alias].get("revision", "main"), local_files_only=True)
        models.append({"alias": alias, "model": model, "revision": Path(path).name, "local_snapshot": path})
    root = Path(__file__).resolve().parents[2]
    save(args.out, {
        "task_id": task, "models": models, "contexts": args.contexts,
        "repetitions": args.repetitions, "output_tokens": args.output_tokens,
        "thinking": False, "temperature": 0,
        "registry_sha256": fingerprint((root / "src/conference_pipeline/model_governance/registry.yaml").read_text()),
        "catalog_sha256": fingerprint((root / "src/conference_pipeline/model_governance/models.yaml").read_text()),
        "scope": "Synthetic source-synthesis speed test; does not establish summary quality.",
    })


def make_prompt(tokenizer, size, salt):
    records = "\n".join(
        f"Source {i}: experiment with {100+i} samples; treatment accuracy {70+i%20} percent; control {60+i%15} percent. "
        "The authors report a small evaluation population, no external replication, and uncertainty under distribution shift. "
        "Summaries must retain attribution, distinguish observations from explanations, and retain limitations."
        for i in range(1, 1000)
    )
    prefix = f"Trial {salt}. Read the source records and produce a detailed numbered synthesis covering findings, comparisons, and limitations.\n"
    source_ids = tokenizer.encode(records)
    length = size - 100
    for _ in range(8):
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": "Synthesize supplied evidence faithfully. Write a detailed answer of at least 1000 words."},
             {"role": "user", "content": prefix + tokenizer.decode(source_ids[:length])}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        ids = tokenizer.encode(prompt)
        delta = size - len(ids)
        if delta == 0:
            break
        length += delta
    return prompt, ids


def worker(args):
    from transformers import AutoTokenizer
    manifest = json.loads(Path(args.plan).read_text())
    entry = next(m for m in manifest["models"] if m["alias"] == args.alias)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    meta = {"engine": args.engine, "model": entry, "model_path": args.model_path,
            "host": platform.node(), "versions": versions(), "prefill_step_size": args.prefill,
            "memory_before": memory(), "manifest": manifest, "rows": []}
    save(output / "results.json", meta)
    model = None
    if args.engine == "direct":
        import mlx.core as mx
        from mlx_lm import load, stream_generate
        from mlx_lm.sample_utils import make_sampler
        start = time.monotonic()
        model, tokenizer = load(args.model_path)
        mx.set_cache_limit(512 * 1024 * 1024)
        meta["load_seconds"] = time.monotonic() - start
        meta["device"] = mx.device_info()
    else:
        import requests

    def generate(prompt, ids, cap):
        start = time.monotonic()
        first = None
        last_time = None
        pieces = []
        usage = None
        finish = None
        native = None
        if model is not None:
            mx.reset_peak_memory()
            last = None
            for last in stream_generate(model, tokenizer, ids, max_tokens=cap,
                                        sampler=make_sampler(temp=0), prefill_step_size=args.prefill):
                now = time.monotonic()
                if first is None:
                    first = now
                last_time = now
                pieces.append(last.text)
            if last is None:
                raise RuntimeError("empty generation")
            count = last.generation_tokens
            native = {"prompt_tps": last.prompt_tps, "generation_tps": last.generation_tps,
                      "peak_memory_gb": last.peak_memory}
            finish = last.finish_reason
            mx.clear_cache()
        else:
            with requests.post(args.url + "/v1/completions", json={
                "model": args.model_path, "prompt": prompt, "temperature": 0,
                "max_tokens": cap, "stream": True, "seed": 0,
                "stream_options": {"include_usage": True},
            }, stream=True, timeout=(10, 240)) as response:
                response.raise_for_status()
                for line in response.iter_lines(chunk_size=1):
                    if not line.startswith(b"data: ") or line == b"data: [DONE]":
                        continue
                    event = json.loads(line[6:])
                    if event.get("error"):
                        raise RuntimeError(str(event["error"]))
                    if event.get("usage"):
                        usage = event["usage"]
                    for choice in event.get("choices", []):
                        text = choice.get("text", "")
                        if text:
                            now = time.monotonic()
                            first = first or now
                            last_time = now
                            pieces.append(text)
                        finish = choice.get("finish_reason") or finish
            count = usage["completion_tokens"] if usage else len(tokenizer.encode("".join(pieces)))
        end = time.monotonic()
        if first is None or last_time is None or count < 2:
            raise RuntimeError("insufficient output to time generation")
        text = "".join(pieces)
        return {"wall_seconds": end - start, "ttft_seconds": first - start,
                "decode_tps": (count - 1) / max(last_time - first, 1e-9),
                "end_to_end_output_tps": count / (end - start), "output_tokens": count,
                "finish_reason": finish, "usage": usage, "native": native,
                "response": text, "response_sha256": fingerprint(text)}

    # Warmup is separate from all timed prompts. Every measured prompt is unique.
    prompt, ids = make_prompt(tokenizer, 256, "warmup")
    meta["warmup"] = generate(prompt, ids, 32)
    save(output / "results.json", meta)
    for size in manifest["contexts"]:
        for trial in range(manifest["repetitions"]):
            prompt, ids = make_prompt(tokenizer, size, trial)
            save(output / f"prompt-{size}-{trial}.json", {"prompt": prompt, "tokens": len(ids), "sha256": fingerprint(prompt)})
            before = memory()
            row = {"requested_prompt_tokens": size, "prompt_tokens": len(ids), "trial": trial,
                   "prompt_sha256": fingerprint(prompt), "memory_before": before}
            try:
                row.update(generate(prompt, ids, manifest["output_tokens"]))
                row["status"] = "succeeded"
            except Exception as error:
                row.update(status="failed", error=str(error))
            row["memory_after"] = memory()
            meta["rows"].append(row)
            save(output / "results.json", meta)
            print(json.dumps({k: v for k, v in row.items() if k not in {"response", "memory_before", "memory_after"}}), flush=True)
    meta["memory_after"] = memory()
    save(output / "results.json", meta)


def report(args):
    root = Path(args.root)
    lines = ["# Local inference speed comparison", "", "Synthetic source-synthesis prompts; thinking disabled; temperature zero. Three trials per context unless marked otherwise. Prefix caching disabled for servers. These measurements do not establish synthesis quality.", "",
             "| Engine | Model alias | Prompt tokens | Trials | Decode tok/s median | TTFT seconds median | End-to-end output tok/s median |", "|---|---|---:|---:|---:|---:|---:|"]
    data = []
    for path in sorted(root.glob("*/results.json")):
        result = json.loads(path.read_text())
        for size in result["manifest"]["contexts"]:
            rows = [r for r in result["rows"] if r["requested_prompt_tokens"] == size and r["status"] == "succeeded"]
            failed = [r for r in result["rows"] if r["requested_prompt_tokens"] == size and r["status"] != "succeeded"]
            if rows:
                med = lambda key: statistics.median(r[key] for r in rows)
                lines.append(f"| {path.parent.name} | {result['model']['alias']} | {int(med('prompt_tokens'))} | {len(rows)} | {med('decode_tps'):.2f} | {med('ttft_seconds'):.2f} | {med('end_to_end_output_tps'):.2f} |")
            if failed:
                lines.append(f"\nFailures for {path.parent.name}, {size}: {failed[0].get('error')}\n")
        data.append(result)
    failures = []
    for status_file in sorted(root.parent.glob("*suite-status.json")):
        failures.extend(r for r in json.loads(status_file.read_text())
                        if r.get("status") == "failed" or r.get("exit_code", 0) != 0)
    if failures:
        lines.extend(["", "## Startup or worker failures", ""])
        for failure in failures:
            lines.append(f"- {failure['profile']} / {failure['alias']}: {failure.get('error', 'worker failed')}")
    lines.extend(["", "## Installed versions", ""])
    environments = {}
    for result in data:
        key = json.dumps(result["versions"], sort_keys=True)
        environments.setdefault(key, []).append(result["model"]["alias"] + ":" + result["engine"] + ":" + str(result["prefill_step_size"]))
    for key, profiles in environments.items():
        lines.append(f"- {', '.join(sorted(set(profiles)))}: `{key}`")
    save(root / "combined-results.json", data)
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="mode", required=True)
    p = subs.add_parser("plan")
    p.add_argument("--aliases", default="qwen36_local,qwen38_local")
    p.add_argument("--contexts", nargs="+", type=int, default=[2048, 16384])
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--output-tokens", type=int, default=256)
    p.add_argument("--out", required=True)
    p = subs.add_parser("worker")
    p.add_argument("--plan", required=True)
    p.add_argument("--alias", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--engine", choices=["direct", "vllm-metal", "vllm-mlx"], required=True)
    p.add_argument("--prefill", type=int, default=512)
    p.add_argument("--url", default="http://127.0.0.1:18081")
    p.add_argument("--out", required=True)
    p = subs.add_parser("report")
    p.add_argument("--root", required=True)
    args = parser.parse_args()
    {"plan": plan, "worker": worker, "report": report}[args.mode](args)


if __name__ == "__main__":
    main()
