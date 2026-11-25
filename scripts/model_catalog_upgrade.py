#!/usr/bin/env python3
"""Propose -- and, only once the operator approves, apply -- model-generation
upgrades to the governed model catalog.

`src/conference_pipeline/model_governance/models.yaml` maps each stable alias
(`opus`, `sonnet`, `gemini_pro`, ...) to one provider/model pair, and every task
in `registry.yaml` refers to an alias. Moving an alias to a newer model is the
one edit a generation upgrade needs -- plus the few places that must agree with
it: the literal `allowed_selections` / benchmark pairs in `registry.yaml` (the
loader checks them against the catalog), the cost model's `task_specs.yaml`,
and a price in `costs/pricing_catalog.json` (a model with no price is costed at
$0 by preflight estimates, which is how unbudgeted spend gets approved).

Three subcommands, used in order by the `model-catalog-upgrade` skill:

  discover  List the models each provider offers today (read-only, $0).
  propose   Compare every active catalog alias with what was discovered and
            print, per alias, the newer same-family model and what moving to
            it would touch. Changes nothing.
  apply     Move ONE approved alias to ONE model. Refuses without a price for
            a model the pricing catalog does not know. Keeps the old model as
            a `status: retired` catalog entry, so reintroducing it fails
            closed at load_registry() time.

Run from the repository root:

    python scripts/model_catalog_upgrade.py discover --out /tmp/discovered.json
    python scripts/model_catalog_upgrade.py propose --discovered /tmp/discovered.json
    python scripts/model_catalog_upgrade.py apply --alias opus --to <model> \\
        --input-price 5 --output-price 25 --price-source anthropic_pricing --dry-run

Provider keys are read from the repository `.env` (ANTHROPIC_API_KEY,
OPENAI_API_KEY, GOOGLE_API_KEY or GEMINI_API_KEY, OPENROUTER_API_KEY); they are
sent only as request headers and never printed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
GOVERNANCE = REPO_ROOT / "src" / "conference_pipeline" / "model_governance"
MODELS_YAML = GOVERNANCE / "models.yaml"
REGISTRY_YAML = GOVERNANCE / "registry.yaml"
TASK_SPECS_YAML = REPO_ROOT / "src" / "conference_pipeline" / "costs" / "task_specs.yaml"
PRICING_JSON = REPO_ROOT / "src" / "conference_pipeline" / "costs" / "pricing_catalog.json"
LITERAL_SCAN_DIRS = ("src/conference_pipeline", "scripts", ".claude/skills")

# Providers whose catalog aliases can be compared against a public model list.
# `local` and NotebookLM models are pinned by runtime receipts, not upgraded here.
DISCOVERABLE = ("anthropic", "openai", "gemini", "openrouter")

_SNAPSHOT_RE = re.compile(r"-(?:\d{8}|\d{4}-\d{2}-\d{2}|\d{2}-\d{2})$")
_UNSTABLE_TOKENS = {"preview", "exp", "experimental", "latest", "beta"}
_VERSION_TOKEN_RE = re.compile(r"^(\d+(?:\.\d+)*)([a-z]*)$")


# --------------------------------------------------------------------------
# model-id parsing
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelShape:
    """A model id split into what it IS (family) and which one (version)."""

    model_id: str
    family: str
    version: tuple[int, ...]
    stable: bool
    snapshot: bool


def model_shape(model_id: str) -> ModelShape:
    """`claude-opus-4-8` -> family claude-opus, version (4, 8). Version tokens
    are dropped from the family (`gpt-X.Y-sol` -> gpt-sol), a `-preview` /
    `-exp` token marks the release unstable, and an `openrouter/<vendor>/`
    route stays in the family so routed and direct models never match."""

    route, _, bare = model_id.rpartition("/")
    snapshot = bool(_SNAPSHOT_RE.search(bare))
    bare = _SNAPSHOT_RE.sub("", bare)
    family_tokens: list[str] = []
    version: list[int] = []
    stable = True
    for token in bare.lower().split("-"):
        if token in _UNSTABLE_TOKENS:
            stable = False
            continue
        match = _VERSION_TOKEN_RE.match(token)
        if match:
            version.extend(int(part) for part in match.group(1).split("."))
            if match.group(2):
                family_tokens.append(match.group(2))
            continue
        if token[:1].isalpha() and token[1:].isdigit():  # "o3" -> family o, version 3
            family_tokens.append(token[0])
            version.append(int(token[1:]))
            continue
        family_tokens.append(token)
    family = "-".join(family_tokens)
    if route:
        family = f"{route}/{family}"
    return ModelShape(model_id, family, tuple(version), stable, snapshot)


def is_newer(candidate: ModelShape, current: ModelShape) -> bool:
    """Same family, and either a higher version or the GA release of a preview."""

    if candidate.family != current.family or candidate.model_id == current.model_id:
        return False
    if candidate.snapshot and not current.snapshot:
        return False  # never trade a moving alias for a dated snapshot
    if candidate.version > current.version:
        return candidate.stable or not current.stable
    return candidate.version == current.version and candidate.stable and not current.stable


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------


def load_env(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _get_json(url: str, headers: dict[str, str]) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "model-catalog-upgrade", **headers})
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - fixed provider hosts
        return json.loads(response.read().decode("utf-8"))


def discover_anthropic() -> list[dict[str, Any]]:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
    models, after = [], ""
    while True:
        page = _get_json(f"https://api.anthropic.com/v1/models?limit=1000{after}", headers)
        models += [{"id": m["id"], "created": m.get("created_at"), "name": m.get("display_name")} for m in page.get("data", [])]
        if not page.get("has_more"):
            return models
        after = f"&after_id={page['last_id']}"


def discover_openai() -> list[dict[str, Any]]:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    page = _get_json("https://api.openai.com/v1/models", {"Authorization": f"Bearer {key}"})
    return [
        {"id": m["id"], "created": dt.datetime.fromtimestamp(m["created"], dt.UTC).date().isoformat() if m.get("created") else None}
        for m in page.get("data", [])
    ]


def discover_gemini() -> list[dict[str, Any]]:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY / GOOGLE_API_KEY not set")
    models, token = [], ""
    while True:
        page = _get_json(
            f"https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000{token}",
            {"x-goog-api-key": key},
        )
        models += [
            {"id": m["name"].removeprefix("models/"), "name": m.get("displayName"),
             "methods": m.get("supportedGenerationMethods", [])}
            for m in page.get("models", [])
        ]
        if not page.get("nextPageToken"):
            return models
        token = f"&pageToken={page['nextPageToken']}"


def discover_openrouter() -> list[dict[str, Any]]:
    headers = {}
    if os.environ.get("OPENROUTER_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['OPENROUTER_API_KEY']}"
    page = _get_json("https://openrouter.ai/api/v1/models", headers)
    models = []
    for m in page.get("data", []):
        pricing = m.get("pricing") or {}
        entry = {"id": f"openrouter/{m['id']}", "name": m.get("name")}
        try:  # OpenRouter prices per token; the catalog records per 1M tokens
            entry["input_per_1m_usd"] = round(float(pricing["prompt"]) * 1e6, 4)
            entry["output_per_1m_usd"] = round(float(pricing["completion"]) * 1e6, 4)
        except (KeyError, TypeError, ValueError):
            pass
        models.append(entry)
    return models


DISCOVERERS = {
    "anthropic": discover_anthropic,
    "openai": discover_openai,
    "gemini": discover_gemini,
    "openrouter": discover_openrouter,
}


def cmd_discover(args: argparse.Namespace) -> int:
    load_env(REPO_ROOT / ".env")
    result: dict[str, Any] = {
        "discovered_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "providers": {},
        "errors": {},
    }
    for provider in args.providers or DISCOVERABLE:
        try:
            result["providers"][provider] = DISCOVERERS[provider]()
        except Exception as exc:  # noqa: BLE001 - one provider down must not hide the rest
            result["errors"][provider] = f"{type(exc).__name__}: {str(exc)[:200]}"
    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        counts = {p: len(v) for p, v in result["providers"].items()}
        print(f"discovered {counts}; errors {result['errors'] or 'none'} -> {args.out}")
    else:
        print(text)
    return 0


# --------------------------------------------------------------------------
# proposal
# --------------------------------------------------------------------------


def load_catalog() -> dict[str, dict[str, Any]]:
    return yaml.safe_load(MODELS_YAML.read_text(encoding="utf-8"))["models"]


def load_registry_tasks() -> dict[str, dict[str, Any]]:
    raw = yaml.safe_load(REGISTRY_YAML.read_text(encoding="utf-8"))
    return raw.get("tasks", raw)


def pricing_entry(provider: str, model: str) -> dict[str, Any] | None:
    catalog = json.loads(PRICING_JSON.read_text(encoding="utf-8"))
    models = (catalog.get("providers", {}).get(provider) or {}).get("models") or {}
    if model in models:
        return models[model]
    aliased = catalog.get("aliases", {}).get(model) or catalog.get("aliases", {}).get(f"{provider}/{model}")
    return models.get(aliased) if aliased else None


def _bounded(literal: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![\w.\-/]){re.escape(literal)}(?![\w.\-])")


def literal_sites(model: str) -> list[str]:
    """Files outside the governed data files that still spell the model out --
    reported for migration, never rewritten (they may be deliberate pins)."""

    pattern = _bounded(model)
    governed = {MODELS_YAML, REGISTRY_YAML, TASK_SPECS_YAML, PRICING_JSON}
    sites = []
    for root in LITERAL_SCAN_DIRS:
        for path in sorted((REPO_ROOT / root).rglob("*")):
            if path.suffix not in {".py", ".md", ".yaml", ".json"} or not path.is_file():
                continue
            if path in governed or path == Path(__file__).resolve() or "tests" in path.parts or path.name.startswith("test_"):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            count = len(pattern.findall(text))
            if count:
                sites.append(f"{path.relative_to(REPO_ROOT)} ({count})")
    return sites


def env_overrides(model: str, env_path: Path | None = None) -> list[str]:
    """`.env` keys whose value is exactly this model (e.g. OCR_MODEL). An env
    override beats the registry default, so after an upgrade it silently keeps
    -- or, once retired, breaks -- the old model. Reports key names only."""

    path = env_path or REPO_ROOT / ".env"
    if not path.is_file():
        return []
    keys = []
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.strip().removeprefix("export ").partition("=")
        if sep and "MODEL" in key.upper() and value.strip().strip("'\"") == model:
            keys.append(key.strip())
    return keys


@dataclass
class Proposal:
    alias: str
    provider: str
    current: str
    candidate: str | None
    candidates: list[str] = field(default_factory=list)
    note: str = ""
    tasks_default: list[str] = field(default_factory=list)
    tasks_fallback: list[str] = field(default_factory=list)
    registry_literals: int = 0
    task_spec_literals: int = 0
    candidate_price: dict[str, Any] | None = None
    suggested_price: dict[str, Any] | None = None
    shared_with: list[str] = field(default_factory=list)
    candidate_alias: str | None = None


def build_proposals(discovered: dict[str, Any], *, include_legacy: bool) -> tuple[list[Proposal], list[str]]:
    catalog = load_catalog()
    tasks = load_registry_tasks()
    registry_text = REGISTRY_YAML.read_text(encoding="utf-8")
    task_specs_text = TASK_SPECS_YAML.read_text(encoding="utf-8")
    offered = {
        provider: {m["id"]: m for m in models}
        for provider, models in discovered.get("providers", {}).items()
    }
    skipped: list[str] = []
    proposals = []
    for alias, entry in catalog.items():
        provider, model = str(entry.get("provider")), str(entry.get("model"))
        if entry.get("status") != "active" or provider not in DISCOVERABLE:
            continue
        if "legacy" in alias and not include_legacy:
            skipped.append(f"{alias} ({model}): legacy pin, skipped (use --include-legacy)")
            continue
        if provider not in offered:
            skipped.append(f"{alias} ({model}): {provider} not discovered ({discovered.get('errors', {}).get(provider, 'not requested')})")
            continue
        current = model_shape(model)
        newer = sorted(
            (model_shape(mid) for mid in offered[provider] if is_newer(model_shape(mid), current)),
            key=lambda s: (s.stable, s.version),
            reverse=True,
        )
        proposal = Proposal(alias=alias, provider=provider, current=model, candidate=newer[0].model_id if newer else None,
                            candidates=[s.model_id for s in newer[:5]])
        if model not in offered[provider]:
            proposal.note = "current model is NOT in the provider's list (renamed or deprecated?)"
        proposal.tasks_default = sorted(t for t, v in tasks.items() if isinstance(v, dict) and v.get("model_ref") == alias)
        proposal.tasks_fallback = sorted(t for t, v in tasks.items() if isinstance(v, dict) and v.get("fallback_model_ref") == alias)
        proposal.registry_literals = len(_bounded(model).findall(registry_text))
        proposal.task_spec_literals = len(_bounded(model).findall(task_specs_text))
        proposal.shared_with = sorted(
            a for a, e in catalog.items()
            if a != alias and e.get("status") == "active" and e.get("provider") == provider and e.get("model") == model
        )
        if proposal.candidate:
            proposal.candidate_alias = next(
                (a for a, e in catalog.items() if e.get("provider") == provider and e.get("model") == proposal.candidate),
                None,
            )
            proposal.candidate_price = pricing_entry(provider, proposal.candidate)
            listed = offered[provider].get(proposal.candidate, {})
            if "input_per_1m_usd" in listed:
                proposal.suggested_price = {k: listed[k] for k in ("input_per_1m_usd", "output_per_1m_usd")}
        proposals.append(proposal)
    return proposals, skipped


def cmd_propose(args: argparse.Namespace) -> int:
    discovered = json.loads(Path(args.discovered).read_text(encoding="utf-8"))
    proposals, skipped = build_proposals(discovered, include_legacy=args.include_legacy)
    upgrades = [p for p in proposals if p.candidate]
    if args.json:
        print(json.dumps({"upgrades": [p.__dict__ for p in upgrades],
                          "current": [p.alias for p in proposals if not p.candidate],
                          "skipped": skipped}, indent=2))
        return 0
    print(f"Catalog: {MODELS_YAML.relative_to(REPO_ROOT)}  (discovered {discovered.get('discovered_at')})\n")
    if not upgrades:
        print("No active alias has a newer same-family model. Nothing to propose.")
    for p in upgrades:
        print(f"## {p.alias} ({p.provider}): {p.current} -> {p.candidate}")
        if len(p.candidates) > 1:
            print(f"   other newer candidates: {', '.join(p.candidates[1:])}")
        if p.note:
            print(f"   NOTE: {p.note}")
        print(f"   default model for {len(p.tasks_default)} task(s); fallback for {len(p.tasks_fallback)}")
        print(f"   literal pairs to rewrite: registry.yaml {p.registry_literals}, task_specs.yaml {p.task_spec_literals}")
        if p.candidate_alias:
            print(f"   ALREADY CATALOGUED as `{p.candidate_alias}`: repoint this alias's tasks to it "
                  f"(model_ref/fallback_model_ref) instead of apply, then retire `{p.alias}`")
        if p.shared_with:
            print(f"   BLOCKER: alias(es) {', '.join(p.shared_with)} also point at {p.current}; apply refuses until resolved")
        if p.candidate_price:
            print(f"   price on file: {p.candidate_price.get('input_per_1m_usd')}/{p.candidate_price.get('output_per_1m_usd')} USD per 1M in/out")
        elif p.suggested_price:
            print(f"   NO price on file; provider lists {p.suggested_price['input_per_1m_usd']}/{p.suggested_price['output_per_1m_usd']} USD per 1M in/out")
        else:
            print("   NO price on file -- look it up on the provider's official pricing page before applying")
        overrides = env_overrides(p.current)
        if overrides:
            print(f"   .env OVERRIDES pin {p.current}: {', '.join(overrides)} -- these beat the registry; "
                  "ask the user to remove or update them")
        sites = literal_sites(p.current)
        if sites:
            print(f"   code/skill sites still spelling {p.current} (report only, not rewritten): {len(sites)}")
            for site in sites[:12]:
                print(f"     - {site}")
        print()
    current = [p.alias for p in proposals if not p.candidate]
    if current:
        print(f"Already current: {', '.join(current)}")
    for line in skipped:
        print(f"Skipped: {line}")
    return 0


# --------------------------------------------------------------------------
# apply
# --------------------------------------------------------------------------


def _retired_alias_name(alias: str, model: str, catalog: dict[str, Any]) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", model.lower().rsplit("/", 1)[-1]).strip("_")
    name = f"{alias}_{slug}_retired"
    n = 2
    while name in catalog:
        name, n = f"{alias}_{slug}_retired_v{n}", n + 1
    return name


def cmd_apply(args: argparse.Namespace) -> int:
    catalog = load_catalog()
    if args.alias not in catalog:
        raise SystemExit(f"unknown alias {args.alias!r}")
    entry = catalog[args.alias]
    provider, old = str(entry["provider"]), str(entry["model"])
    new = args.to
    if entry.get("status") != "active":
        raise SystemExit(f"{args.alias} is {entry.get('status')}; only active aliases are upgraded")
    if old == new:
        raise SystemExit(f"{args.alias} already points at {new}")
    if any(e.get("model") == new and e.get("provider") == provider for e in catalog.values()):
        clash = [a for a, e in catalog.items() if e.get("model") == new and e.get("provider") == provider]
        raise SystemExit(f"{provider}/{new} is already catalogued as {clash}; point tasks at that alias instead")
    shared = [a for a, e in catalog.items() if a != args.alias and e.get("status") == "active"
              and e.get("provider") == provider and e.get("model") == old]
    if shared:
        raise SystemExit(f"{', '.join(shared)} also use {old}; upgrading {args.alias} alone would leave "
                         "registry pairs ambiguous -- decide those aliases first")
    price = pricing_entry(provider, new)
    if price is None and (args.input_price is None or args.output_price is None or not args.price_source):
        raise SystemExit(f"no price on file for {provider}/{new}: pass --input-price, --output-price and "
                         "--price-source (a key of pricing_catalog.json 'sources', or a URL). An unpriced "
                         "model is costed at $0 by preflight estimates.")

    today = dt.date.today().isoformat()
    changes: list[str] = []

    # 1. models.yaml: move the alias, keep the old pair as a retired entry.
    models_text = MODELS_YAML.read_text(encoding="utf-8")
    block = re.search(rf"(?m)^  {re.escape(args.alias)}:\n(?:    .*\n|\s*\n)*?", models_text)
    model_line = re.compile(rf"(?m)^(    model: ){re.escape(old)}\s*$")
    start = block.start() if block else -1
    match = model_line.search(models_text, start) if block else None
    next_alias = re.compile(r"(?m)^  [A-Za-z0-9_]+:\s*$").search(models_text, start + 1) if block else None
    if match and next_alias and match.start() > next_alias.start():
        match = None  # the old model line belongs to a later alias, not this one
    if not match:
        raise SystemExit(f"could not locate `model: {old}` under `{args.alias}:` in models.yaml")
    models_text = models_text[: match.start()] + f"    model: {new}" + models_text[match.end():]
    retired = _retired_alias_name(args.alias, old, catalog)
    models_text = models_text.rstrip("\n") + (
        f"\n  {retired}:\n    provider: {provider}\n    model: {old}\n    status: retired\n"
        f"    notes: \"Superseded by {new} when alias `{args.alias}` was upgraded on {today} "
        f"(model-catalog-upgrade, operator-approved).\"\n"
    )
    changes.append(f"models.yaml: {args.alias} {old} -> {new}; {old} kept as retired alias {retired}")

    # 2. registry.yaml / task_specs.yaml: literal pairs must name the new model
    #    (the loader rejects a retired pair in allowed_selections).
    edits = {}
    for path in (REGISTRY_YAML, TASK_SPECS_YAML):
        text = path.read_text(encoding="utf-8")
        updated, count = _bounded(old).subn(new, text)
        edits[path] = updated
        changes.append(f"{path.name}: {count} literal(s) {old} -> {new}")

    # 3. pricing: never leave the new model unpriced.
    pricing = json.loads(PRICING_JSON.read_text(encoding="utf-8"))
    if price is None:
        pricing["providers"].setdefault(provider, {}).setdefault("models", {})[new] = {
            "pricing_unit": "tokens",
            "input_per_1m_usd": float(args.input_price),
            "output_per_1m_usd": float(args.output_price),
            "source": args.price_source,
            "notes": f"Added {today} by model-catalog-upgrade when alias `{args.alias}` moved from {old}.",
        }
        changes.append(f"pricing_catalog.json: {provider}/{new} priced {args.input_price}/{args.output_price} USD per 1M ({args.price_source})")
    else:
        changes.append(f"pricing_catalog.json: {provider}/{new} already priced; unchanged")

    print("\n".join(("DRY RUN -- would change:" if args.dry_run else "Changed:", *(f"  - {c}" for c in changes))))
    if args.dry_run:
        return 0
    MODELS_YAML.write_text(models_text + "\n", encoding="utf-8")
    for path, text in edits.items():
        path.write_text(text, encoding="utf-8")
    PRICING_JSON.write_text(json.dumps(pricing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    check = subprocess.run(
        [sys.executable, "-c",
         "from conference_pipeline.model_governance.registry import load_registry; load_registry(); print('load_registry OK')"],
        cwd=REPO_ROOT, env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        capture_output=True, text=True,
    )
    print(check.stdout.strip() or check.stderr.strip()[-800:])
    overrides = env_overrides(old)
    if overrides:
        print(f"\n.env still pins the retired {old} via {', '.join(overrides)}: ask the user to remove or update "
              "these lines, or the tasks that read them will refuse to run.")
    sites = literal_sites(old)
    if sites:
        print(f"\n{len(sites)} code/skill site(s) still spell {old} -- migrate or mark `# model-literal-ok: <reason>`:")
        for site in sites:
            print(f"  - {site}")
    return 0 if check.returncode == 0 else 1


def _run_load_registry() -> int:
    check = subprocess.run(
        [sys.executable, "-c",
         "from conference_pipeline.model_governance.registry import load_registry; load_registry(); print('load_registry OK')"],
        cwd=REPO_ROOT, env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        capture_output=True, text=True,
    )
    print(check.stdout.strip() or check.stderr.strip()[-800:])
    return check.returncode


def cmd_repoint(args: argparse.Namespace) -> int:
    """Move every task off one alias onto another, then retire the first --
    for an alias whose model is gone, or whose successor is already catalogued."""

    catalog = load_catalog()
    for alias in (args.alias, args.into):
        if alias not in catalog:
            raise SystemExit(f"unknown alias {alias!r}")
    old, new = catalog[args.alias], catalog[args.into]
    if new.get("status") != "active":
        raise SystemExit(f"{args.into} is {new.get('status')}; repoint only onto an active alias")
    if old.get("provider") != new.get("provider"):
        raise SystemExit(f"{args.alias} is {old.get('provider')} but {args.into} is {new.get('provider')}; "
                         "a repoint must keep the task's provider route")
    old_model, new_model = str(old["model"]), str(new["model"])
    shared = [a for a, e in catalog.items() if a != args.alias and e.get("status") == "active"
              and e.get("provider") == old.get("provider") and e.get("model") == old_model]
    if shared:
        raise SystemExit(f"{', '.join(shared)} also use {old_model}; resolve those first")
    if pricing_entry(str(new["provider"]), new_model) is None:
        raise SystemExit(f"{new['provider']}/{new_model} has no price on file; run `price` first")

    changes = []
    registry_text = REGISTRY_YAML.read_text(encoding="utf-8")
    ref_re = re.compile(rf"(?m)^(\s*(?:model_ref|fallback_model_ref):\s*){re.escape(args.alias)}\s*$")
    registry_text, refs = ref_re.subn(rf"\g<1>{args.into}", registry_text)
    registry_text, literals = _bounded(old_model).subn(new_model, registry_text)
    changes.append(f"registry.yaml: {refs} model_ref/fallback_model_ref {args.alias} -> {args.into}; "
                   f"{literals} literal(s) {old_model} -> {new_model}")
    specs_text, spec_literals = _bounded(old_model).subn(new_model, TASK_SPECS_YAML.read_text(encoding="utf-8"))
    changes.append(f"task_specs.yaml: {spec_literals} literal(s) {old_model} -> {new_model}")

    models_text = MODELS_YAML.read_text(encoding="utf-8")
    block = re.search(rf"(?m)^  {re.escape(args.alias)}:\n", models_text)
    status = re.compile(r"(?m)^    status: active\s*$").search(models_text, block.end()) if block else None
    next_alias = re.compile(r"(?m)^  [A-Za-z0-9_]+:\s*$").search(models_text, block.end()) if block else None
    if not status or (next_alias and status.start() > next_alias.start()):
        raise SystemExit(f"could not locate `status: active` under `{args.alias}:` in models.yaml")
    models_text = models_text[: status.start()] + "    status: retired" + models_text[status.end():]
    changes.append(f"models.yaml: {args.alias} ({old_model}) retired; its tasks now use {args.into} ({new_model})")

    print("\n".join(("DRY RUN -- would change:" if args.dry_run else "Changed:", *(f"  - {c}" for c in changes))))
    if args.dry_run:
        return 0
    REGISTRY_YAML.write_text(registry_text, encoding="utf-8")
    TASK_SPECS_YAML.write_text(specs_text, encoding="utf-8")
    MODELS_YAML.write_text(models_text, encoding="utf-8")
    return _run_load_registry()


def cmd_price(args: argparse.Namespace) -> int:
    """Set or refresh one model's price from an official source."""

    pricing = json.loads(PRICING_JSON.read_text(encoding="utf-8"))
    models = pricing["providers"].setdefault(args.provider, {}).setdefault("models", {})
    before = models.get(args.model)
    entry = dict(before or {"pricing_unit": "tokens"})
    entry.update({
        "input_per_1m_usd": float(args.input_price),
        "output_per_1m_usd": float(args.output_price),
        "source": args.price_source,
    })
    if args.note:
        entry["notes"] = args.note
    was = f"{before.get('input_per_1m_usd')}/{before.get('output_per_1m_usd')}" if before else "unpriced"
    print(f"{'DRY RUN -- would set' if args.dry_run else 'Set'} {args.provider}/{args.model}: "
          f"{was} -> {args.input_price}/{args.output_price} USD per 1M ({args.price_source})")
    if args.dry_run:
        return 0
    models[args.model] = entry
    PRICING_JSON.write_text(json.dumps(pricing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("discover", help="list the models each provider offers today ($0)")
    d.add_argument("--providers", nargs="*", choices=DISCOVERABLE)
    d.add_argument("--out")
    d.set_defaults(func=cmd_discover)
    p = sub.add_parser("propose", help="compare the catalog with discovered models; changes nothing")
    p.add_argument("--discovered", required=True)
    p.add_argument("--include-legacy", action="store_true", help="also consider aliases named *legacy*")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_propose)
    a = sub.add_parser("apply", help="move ONE operator-approved alias to ONE model")
    a.add_argument("--alias", required=True)
    a.add_argument("--to", required=True)
    a.add_argument("--input-price", type=float)
    a.add_argument("--output-price", type=float)
    a.add_argument("--price-source")
    a.add_argument("--dry-run", action="store_true")
    a.set_defaults(func=cmd_apply)
    r = sub.add_parser("repoint", help="move ONE approved alias's tasks onto another alias, then retire it")
    r.add_argument("--alias", required=True, help="the alias to retire")
    r.add_argument("--into", required=True, help="the active alias its tasks move to")
    r.add_argument("--dry-run", action="store_true")
    r.set_defaults(func=cmd_repoint)
    pr = sub.add_parser("price", help="set or refresh one model's price from an official source")
    pr.add_argument("--provider", required=True)
    pr.add_argument("--model", required=True)
    pr.add_argument("--input-price", type=float, required=True)
    pr.add_argument("--output-price", type=float, required=True)
    pr.add_argument("--price-source", required=True)
    pr.add_argument("--note")
    pr.add_argument("--dry-run", action="store_true")
    pr.set_defaults(func=cmd_price)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
