"""Per-deck step lineage: input-digest binding for presentation steps.

The presentation stage runs a chain of per-deck steps (source inventory,
blueprint, deck contract, figures, render, narration, review, video). On
``--resume`` a step used to be skipped whenever its output file existed, so an
edited blueprint or a patched ``engaging_narration_raw.json`` silently reused
stale slides and videos. This module replaces that with a ledger per deck,
``<deck_dir>/step_lineage.json``, recording for every step the digests of the
inputs it read and the outputs it wrote. A step is skipped only when all of
that still matches.

Two rules make the chain converge instead of churning:

* **Writer history.** Some steps edit files in place that earlier steps
  produced and later steps consume (bookend slides rewrite ``render_spec.json``;
  ``reconcile_contract_hashes`` re-stamps the delivery contracts; the
  ``fix_*`` narration steps rewrite the script they were given). A path's
  history lists ``(step, sha256)`` writes; a producer resets it, an
  ``in_place`` editor appends. A reader records *which* write it read. A reader
  that ran before the editor still finds its entry (only in-place edits
  followed) and skips; a reader that ran after the editor re-runs only when
  the editor's output actually changed.
* **Adoption.** A few outputs are sanctioned operator-edit surfaces
  (``engaging_narration_raw.json``, the ENGAGING markdowns, a repaired
  blueprint). When such a file no longer matches its recorded digest, the
  ledger adopts the edit as the new head instead of re-running the producer;
  every consumer that recorded the old digest re-runs. Once adopted, the
  producer is pinned: it stays skipped even if one of *its own* inputs
  changes afterwards (an unrelated upstream regeneration must never silently
  discard an operator's repair), for as long as every write since the
  adoption is an in-place edit rather than a fresh reproduction. A JSON
  output carrying a non-empty ``repair_of`` is pinned the same way, so the
  protection survives a ledger rebuilt from disk. Only deleting/emptying the
  output, or a reproduction without the marker, reopens it.

Reuse precedent: ``presentation.vision_qc`` already reuses a per-slide verdict
only when ``screenshot_sha256`` still matches; this generalises that rule to
every step and every input.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from conference_pipeline.receipt_lineage import (
    canonical_json_digest,
    directory_sha256,
    sha256_file,
)
from conference_pipeline.workflow import PROJECT_ROOT

LEDGER_FILENAME = "step_lineage.json"
LEDGER_SCHEMA_VERSION = 1
DEFAULT_REJECT_STATUSES = frozenset({"blocked", "failed", "halted", "error"})
# Files at or above this size reuse the recorded digest when size and mtime
# match (git-index style), so re-hashing videos and PDFs does not dominate a
# resume. Small JSON/Markdown contracts are always re-hashed: a same-size,
# same-second hand edit of a contract must never be missed.
LARGE_FILE_STAT_CACHE_BYTES = 8 * 1024 * 1024
DEFAULT_IGNORED_FLAGS = frozenset(
    {"--workspace", "--deck-dir", "--build-dir", "--generated-dir", "--run-id", "--cache-dir"}
)
_SESSION_DIGEST_CACHE: dict[tuple[str, int, int], str] = {}
_INTERPRETER = Path(sys.executable).resolve()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _carries_repair_marker(path: Path) -> bool:
    if path.suffix != ".json" or not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, Mapping) and bool(payload.get("repair_of"))


@dataclass(frozen=True)
class StepInput:
    """One declared input: a file, or a directory (optionally filtered by glob)."""

    path: Path
    pattern: str | None = None
    optional: bool = True


@dataclass(frozen=True)
class SkipDecision:
    skip: bool
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepSpec:
    """A step's identity, inputs, outputs and skip policy."""

    step_id: str
    inputs: tuple[StepInput, ...]
    outputs: tuple[Path, ...]
    tracked_outputs: tuple[StepInput, ...] = ()
    editable_outputs: frozenset[Path] = frozenset()
    in_place: bool = False
    values: Mapping[str, Any] = field(default_factory=dict)
    accept_statuses: frozenset[str] | None = None
    reject_statuses: frozenset[str] = DEFAULT_REJECT_STATUSES
    command: tuple[str, ...] = ()

    @classmethod
    def from_command(
        cls,
        command: Sequence[str],
        *,
        workspace: Path,
        outputs: Iterable[Path],
        step_id: str | None = None,
        output_flags: Iterable[str] = (),
        input_flags: Iterable[str] = (),
        extra_inputs: Iterable[StepInput | Path] = (),
        directory_patterns: Mapping[Path, str] | None = None,
        ignored_flags: Iterable[str] = DEFAULT_IGNORED_FLAGS,
        ignored_paths: Iterable[Path] = (),
        repo_root: Path = PROJECT_ROOT,
        **overrides: Any,
    ) -> "StepSpec":
        """Derive inputs from argv: every value that resolves to an existing
        file or directory under the workspace or repo root and is not one of
        the declared or tracked outputs. Non-path flag values become
        ``values`` so a changed switch re-runs the step. ``extra_inputs``
        covers reads that never appear on argv (``--deck-dir`` implicit
        reads, brand assets); ``directory_patterns`` narrows an argv
        directory input to a glob (``assets/generated`` → ``*.svg``) so later
        writes beside it do not churn the step."""

        argv = [str(token) for token in command]
        output_paths = tuple(Path(item).resolve() for item in outputs)
        tracked = tuple(
            item if isinstance(item, StepInput) else StepInput(Path(item))
            for item in overrides.pop("tracked_outputs", ())
        )
        output_set = (
            set(output_paths)
            | {item.path.resolve() for item in tracked}
            | {Path(item).resolve() for item in ignored_paths}
        )
        output_flag_set = set(output_flags)
        input_flag_set = set(input_flags)
        ignored = set(ignored_flags)
        workspace_root = Path(workspace).resolve()
        repo = Path(repo_root).resolve()
        patterns = {Path(key).resolve(): value for key, value in (directory_patterns or {}).items()}

        def _inside(resolved: Path) -> bool:
            for base in (workspace_root, repo):
                try:
                    resolved.relative_to(base)
                    return True
                except ValueError:
                    continue
            return False

        def _candidate(token: str) -> tuple[Path, bool] | str | None:
            """``(path, exists)`` for a token that names a file under the
            workspace or repo; ``"ignored"`` for one of this step's outputs,
            tracked outputs or ignored paths (neither an input nor a value);
            ``None`` for a non-path token. An absolute path that does not
            exist yet is still an input (recorded as absent) rather than a
            value: whether it exists must never change how the step is keyed."""

            if not token or token.startswith("-"):
                return None
            path = Path(token)
            if not path.is_absolute():
                for base in (workspace_root, repo, Path.cwd()):
                    try:
                        found = (base / path).exists()
                    except OSError:
                        # A value the OS cannot even stat (a comma-joined list
                        # of slide ids over the name-length limit) is not a path.
                        return None
                    if found:
                        path = base / path
                        break
                else:
                    return None
            resolved = path.resolve()
            if resolved == _INTERPRETER:
                return "ignored"
            if resolved in output_set or any(out in resolved.parents for out in output_set):
                return "ignored"
            if not _inside(resolved):
                return None
            return resolved, resolved.exists()

        inputs: list[StepInput] = []
        seen: set[Path] = set()
        values: dict[str, Any] = {}
        script_name: str | None = None
        index = 0
        while index < len(argv):
            token = argv[index]
            if script_name is None and token.endswith(".py"):
                script_name = Path(token).stem
            if token.startswith("--"):
                flag, eq, inline = token.partition("=")
                value: str | None
                consumed = 0
                if eq:
                    value = inline
                elif index + 1 < len(argv) and not argv[index + 1].startswith("--"):
                    value = argv[index + 1]
                    consumed = 1
                else:
                    value = None
                if flag in ignored or flag in output_flag_set:
                    index += 1 + consumed
                    continue
                if value is None:
                    values[flag] = True
                    index += 1
                    continue
                candidate = _candidate(value)
                if candidate == "ignored":
                    index += 1 + consumed
                    continue
                if candidate is not None:
                    resolved, exists = candidate
                    if resolved not in seen:
                        seen.add(resolved)
                        inputs.append(
                            StepInput(resolved, pattern=patterns.get(resolved), optional=not exists)
                        )
                elif flag in input_flag_set:
                    inputs.append(StepInput(Path(value), optional=True))
                else:
                    previous = values.get(flag)
                    if previous is None:
                        values[flag] = value
                    elif isinstance(previous, list):
                        previous.append(value)
                    else:
                        values[flag] = [previous, value]
                index += 1 + consumed
                continue
            candidate = _candidate(token)
            if (
                candidate is not None
                and candidate != "ignored"
                and candidate[0] not in seen
                and script_name != Path(token).stem
            ):
                resolved, exists = candidate
                seen.add(resolved)
                inputs.append(StepInput(resolved, pattern=patterns.get(resolved), optional=not exists))
            index += 1

        for extra in extra_inputs:
            item = extra if isinstance(extra, StepInput) else StepInput(Path(extra))
            resolved = item.path if item.path.is_absolute() else workspace_root / item.path
            if resolved.resolve() in seen:
                continue
            seen.add(resolved.resolve())
            inputs.append(StepInput(resolved, pattern=item.pattern, optional=item.optional))

        merged_values = dict(values)
        merged_values.update(dict(overrides.pop("values", {}) or {}))
        editable = frozenset(Path(item).resolve() for item in overrides.pop("editable_outputs", ()))
        return cls(
            step_id=step_id or script_name or "step",
            inputs=tuple(inputs),
            outputs=output_paths,
            tracked_outputs=tracked,
            editable_outputs=editable,
            values=merged_values,
            command=tuple(argv),
            **overrides,
        )


class DeckStepLineage:
    """The per-deck ledger. One instance per deck directory per run."""

    def __init__(self, ledger_path: Path, *, workspace: Path, repo_root: Path = PROJECT_ROOT) -> None:
        self.ledger_path = Path(ledger_path)
        self.workspace = Path(workspace).resolve()
        self.repo_root = Path(repo_root).resolve()
        self._data = self._load()
        self._rerun_reasons: dict[str, str] = {}
        self._adopted_this_session: set[str] = set()

    @classmethod
    def for_deck_dir(cls, deck_dir: Path, *, workspace: Path, repo_root: Path = PROJECT_ROOT) -> "DeckStepLineage":
        return cls(Path(deck_dir) / LEDGER_FILENAME, workspace=workspace, repo_root=repo_root)

    # -- persistence --------------------------------------------------------

    def _load(self) -> dict[str, Any]:
        empty = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "deck_dir": self._key(self.ledger_path.parent),
            "steps": {},
            "paths": {},
        }
        if not self.ledger_path.is_file():
            return empty
        try:
            loaded = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return empty
        if not isinstance(loaded, Mapping) or loaded.get("schema_version") != LEDGER_SCHEMA_VERSION:
            return empty
        data = dict(loaded)
        data.setdefault("steps", {})
        data.setdefault("paths", {})
        return data

    def save(self) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.ledger_path.with_name(f".{self.ledger_path.name}.{os.getpid()}.tmp")
        temp.write_text(json.dumps(self._data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temp, self.ledger_path)

    # -- keys and digests ---------------------------------------------------

    def _key(self, path: Path) -> str:
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = self.workspace / resolved
        resolved = resolved.resolve()
        for prefix, base in (("ws", self.workspace), ("repo", self.repo_root)):
            try:
                return f"{prefix}:{resolved.relative_to(base).as_posix()}"
            except ValueError:
                continue
        return f"abs:{resolved.as_posix()}"

    def _resolve(self, key: str) -> Path:
        prefix, _, rest = key.partition(":")
        if prefix == "ws":
            return self.workspace / rest
        if prefix == "repo":
            return self.repo_root / rest
        return Path(rest)

    @staticmethod
    def _file_digest(path: Path, recorded: Mapping[str, Any] | None = None) -> tuple[str, int, int]:
        stat = path.stat()
        cache_key = (str(path), stat.st_size, stat.st_mtime_ns)
        cached = _SESSION_DIGEST_CACHE.get(cache_key)
        if cached is not None:
            return cached, stat.st_size, stat.st_mtime_ns
        if (
            stat.st_size >= LARGE_FILE_STAT_CACHE_BYTES
            and recorded is not None
            and recorded.get("bytes") == stat.st_size
            and recorded.get("mtime_ns") == stat.st_mtime_ns
            and isinstance(recorded.get("sha256"), str)
        ):
            digest = str(recorded["sha256"])
        else:
            digest = sha256_file(path)
        _SESSION_DIGEST_CACHE[cache_key] = digest
        return digest, stat.st_size, stat.st_mtime_ns

    def _observe(self, path: Path, *, pattern: str | None = None, recorded: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if path.is_file():
            digest, size, mtime_ns = self._file_digest(path, recorded)
            return {"exists": True, "kind": "file", "sha256": digest, "bytes": size, "mtime_ns": mtime_ns}
        if path.is_dir():
            # A directory is its matching files: one with no match (yet) is the
            # same as one that does not exist, so a sibling step creating the
            # directory for its own files does not read as this step's output
            # or input having appeared.
            iterator = path.rglob(pattern) if pattern else path.rglob("*")
            if not any(candidate.is_file() for candidate in iterator):
                return {"exists": False, "kind": "dir", "sha256": None, "pattern": pattern}
            return {"exists": True, "kind": "dir", "sha256": directory_sha256(path, pattern=pattern), "pattern": pattern}
        record: dict[str, Any] = {"exists": False, "kind": None, "sha256": None}
        if pattern:
            record["pattern"] = pattern
        return record

    def _history(self, key: str) -> list[dict[str, Any]]:
        entry = self._data["paths"].get(key)
        history = entry.get("history") if isinstance(entry, Mapping) else None
        return list(history) if isinstance(history, list) else []

    def _head_sha(self, key: str) -> str | None:
        history = self._history(key)
        return str(history[-1].get("sha256")) if history else None

    # -- the contract -------------------------------------------------------

    def snapshot_inputs(self, step: StepSpec) -> dict[str, Any]:
        """Digest every input *before* the step runs; records which write of a
        tracked path the step is about to read."""

        inputs: dict[str, Any] = {}
        for item in step.inputs:
            key = self._key(item.path)
            observed = self._observe(self._resolve(key), pattern=item.pattern)
            observed["optional"] = item.optional
            history = self._history(key)
            if history and observed.get("sha256") == history[-1].get("sha256"):
                observed["writer"] = history[-1].get("step")
            inputs[key] = observed
        return {
            "inputs": inputs,
            "values_sha256": canonical_json_digest(dict(step.values)),
            "taken_at": _utc_now_iso(),
        }

    def record(self, step: StepSpec, snapshot: Mapping[str, Any], *, status: str = "ran") -> None:
        """Record a successful run: digest outputs, update path histories."""

        outputs: dict[str, Any] = {}
        for path, required in [(out, True) for out in step.outputs] + [
            (item.path, False) for item in step.tracked_outputs
        ]:
            key = self._key(path)
            pattern = next((item.pattern for item in step.tracked_outputs if item.path == path), None)
            observed = self._observe(self._resolve(key), pattern=pattern)
            observed["required"] = required
            outputs[key] = observed
            if not observed["exists"]:
                continue
            new_sha = str(observed["sha256"])
            history = self._history(key)
            if step.in_place and history:
                # A re-run of an in-place editor supersedes its own earlier
                # write and everything after it: readers that read the earlier
                # write must find it gone, not "followed only by in-place edits".
                own = next(
                    (i for i in range(len(history) - 1, -1, -1) if history[i].get("step") == step.step_id),
                    None,
                )
                if own is not None:
                    history = history[:own]
                if not history or history[-1].get("sha256") != new_sha:
                    history.append({"step": step.step_id, "sha256": new_sha, "in_place": True})
            else:
                history = [{"step": step.step_id, "sha256": new_sha}]
            self._data["paths"][key] = {"history": history, "pattern": pattern}
        now = _utc_now_iso()
        entry: dict[str, Any] = {
            "last_status": status,
            "last_run_at": now,
            "last_visit_at": now,
            "inputs": dict(snapshot.get("inputs") or {}),
            "values_sha256": snapshot.get("values_sha256"),
            "outputs": outputs,
            "in_place": step.in_place,
            "command": list(step.command),
        }
        rerun_reason = self._rerun_reasons.pop(step.step_id, None)
        if rerun_reason is not None:
            entry["last_decision"] = rerun_reason
        self._data["steps"][step.step_id] = entry
        self.save()

    def note_skip(self, step: StepSpec, decision: SkipDecision) -> None:
        entry = self._data["steps"].get(step.step_id)
        if isinstance(entry, dict):
            entry["last_visit_at"] = _utc_now_iso()
            if step.step_id not in self._adopted_this_session:
                entry["last_status"] = "reused"
            entry["last_decision"] = decision.reason
            self.save()

    def note_rerun(self, step: StepSpec, decision: SkipDecision) -> None:
        """Remember why a resumed run re-ran a step (surfaced in receipts)."""

        self._rerun_reasons[step.step_id] = decision.reason
        entry = self._data["steps"].get(step.step_id)
        if isinstance(entry, dict):
            entry["last_decision"] = decision.reason
            entry["last_decision_details"] = dict(decision.details)
            self.save()

    def adopt(self, step: StepSpec) -> bool:
        """Bootstrap a ledger entry from current disk state (migration only)."""

        if step.step_id in self._data["steps"]:
            return False
        for out in step.outputs:
            if not out.is_file() or out.stat().st_size == 0:
                if not out.is_dir():
                    return False
        self.record(step, self.snapshot_inputs(step), status="adopted")
        self._adopted_this_session.add(step.step_id)
        return True

    def should_skip(self, step: StepSpec, *, resume: bool) -> SkipDecision:
        if not resume:
            return SkipDecision(False, "resume_disabled")
        entry = self._data["steps"].get(step.step_id)
        if not isinstance(entry, Mapping):
            return SkipDecision(False, "no_sidecar")

        for out in step.outputs:
            if out.is_dir():
                continue
            if not out.is_file() or out.stat().st_size == 0:
                return SkipDecision(False, f"outputs_missing:{self._key(out)}")

        for out in step.outputs:
            if out.suffix != ".json" or not out.is_file():
                continue
            try:
                payload = json.loads(out.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return SkipDecision(False, f"output_unreadable:{self._key(out)}")
            status = payload.get("status") if isinstance(payload, Mapping) else None
            if isinstance(status, str):
                if status in step.reject_statuses:
                    return SkipDecision(False, f"status_rejected:{self._key(out)}", {"status": status})
                if step.accept_statuses is not None and status not in step.accept_statuses:
                    return SkipDecision(False, f"status_not_accepted:{self._key(out)}", {"status": status})

        editable_keys = {self._key(path) for path in step.editable_outputs}
        adopted: list[str] = []
        recorded_outputs = entry.get("outputs") if isinstance(entry.get("outputs"), Mapping) else {}
        for key, rec in recorded_outputs.items():
            rec = rec if isinstance(rec, Mapping) else {}
            observed = self._observe(self._resolve(key), pattern=rec.get("pattern"), recorded=rec)
            if not observed["exists"]:
                if rec.get("exists"):
                    return SkipDecision(False, f"output_changed:{key}", {"observed": "<missing>"})
                continue
            head = self._head_sha(key) or rec.get("sha256")
            if observed["sha256"] != head:
                if key in editable_keys:
                    self._data["paths"][key] = {
                        "history": [
                            {"step": step.step_id, "sha256": observed["sha256"], "adopted_operator_edit": True}
                        ]
                    }
                    rec_copy = dict(rec)
                    rec_copy.update({k: v for k, v in observed.items() if k in ("sha256", "bytes", "mtime_ns")})
                    self._data["steps"][step.step_id]["outputs"][key] = rec_copy
                    adopted.append(key)
                    continue
                return SkipDecision(False, f"output_changed:{key}", {"recorded": head, "observed": observed["sha256"]})

        # An in-place editor's own write is its "read position" on the paths it
        # edits: it stays current while that write is still in the history and
        # only in-place edits followed; a re-produced file re-runs the editor.
        # Checked on the outputs themselves, because argv-derived inputs never
        # include a step's own outputs.
        self_edited = {self._key(out) for out in step.outputs} if step.in_place else set()
        for key in sorted(self_edited - set(adopted)):
            rec = recorded_outputs.get(key)
            if not isinstance(rec, Mapping) or not rec.get("exists"):
                continue
            history = self._history(key)
            if not history:
                continue
            own_sha = rec.get("sha256")
            position = next(
                (i for i, write in enumerate(history) if write.get("sha256") == own_sha and write.get("step") == step.step_id),
                None,
            )
            if position is None:
                # A byte-identical in-place write never appended its own entry.
                position = next((i for i, write in enumerate(history) if write.get("sha256") == own_sha), None)
            if position is None:
                return SkipDecision(False, f"output_changed:{key}", {"reason": "write_superseded"})
            if any(not write.get("in_place") for write in history[position + 1 :]):
                return SkipDecision(False, f"output_changed:{key}", {"reason": "reproduced"})

        # An operator-edited output (a versioned repair, adopted above or in an
        # earlier resume) is authoritative once adopted: an upstream input
        # drifting afterwards -- an unrelated regeneration two steps back that
        # happens to feed this one -- must never force the producer to re-run
        # and silently overwrite the repair with a fresh, unreviewed
        # generation. The adopted write need not be the head: a downstream
        # in-place editor (e.g. narration's ``#post-gates --sync-raw-from-md``)
        # legitimately appends after it the same way it appends after a
        # producer's own write, so the pin holds while every write since the
        # adoption is ``in_place`` -- mirroring the rule just above. Only a
        # missing/emptied output (checked above), a further hand edit (which
        # replaces history with a fresh adopted head), or a non-in-place
        # reproduction without a repair marker reopens this step.
        def _pinned(key: str) -> bool:
            history = self._history(key)
            position = next(
                (i for i in range(len(history) - 1, -1, -1) if history[i].get("adopted_operator_edit")),
                None,
            )
            if position is not None and all(write.get("in_place") for write in history[position + 1 :]):
                return True
            # The ledger flag is lost whenever the ledger is rebuilt from disk
            # (--adopt-existing-outputs), so the artifact's own repair marker
            # also pins it: a versioned repair records ``repair_of``, a fresh
            # generation does not.
            return _carries_repair_marker(self._resolve(key))

        pinned_keys = sorted(key for key in editable_keys if _pinned(key))
        if pinned_keys:
            if adopted:
                self.save()
            return SkipDecision(True, "pinned_by_operator_edit", {"adopted": adopted, "pinned": pinned_keys})

        recorded_inputs = entry.get("inputs") if isinstance(entry.get("inputs"), Mapping) else {}
        declared_keys = {self._key(item.path) for item in step.inputs}
        if declared_keys != set(recorded_inputs):
            # The step now reads a different set of files than it did (an
            # argv input added or dropped): nothing recorded vouches for it.
            return SkipDecision(
                False,
                "inputs_changed:declared",
                {
                    "added": sorted(declared_keys - set(recorded_inputs)),
                    "removed": sorted(set(recorded_inputs) - declared_keys),
                },
            )
        for key, rec in recorded_inputs.items():
            rec = rec if isinstance(rec, Mapping) else {}
            if key in self_edited:
                own_output = recorded_outputs.get(key)
                rec = {
                    **rec,
                    "writer": step.step_id,
                    "sha256": (own_output or {}).get("sha256") if isinstance(own_output, Mapping) else None,
                    "exists": bool((own_output or {}).get("exists")) if isinstance(own_output, Mapping) else False,
                }
            observed = self._observe(self._resolve(key), pattern=rec.get("pattern"), recorded=rec)
            if bool(observed["exists"]) != bool(rec.get("exists")):
                return SkipDecision(False, f"input_changed:{key}", {"existence": observed["exists"]})
            if not observed["exists"]:
                continue
            history = self._history(key)
            if history:
                head = history[-1].get("sha256")
                if observed["sha256"] != head:
                    return SkipDecision(False, f"input_changed:{key}", {"reason": "hand_edit", "head": head})
                writer = rec.get("writer")
                recorded_sha = rec.get("sha256")
                # A self-edited path matches by digest alone: an in-place write
                # that left the bytes unchanged never appended its own entry.
                match_writer = None if key in self_edited else writer
                position = next(
                    (
                        i
                        for i, write in enumerate(history)
                        if write.get("sha256") == recorded_sha
                        and (match_writer is None or write.get("step") == match_writer)
                    ),
                    None,
                )
                if position is None:
                    return SkipDecision(False, f"input_changed:{key}", {"reason": "write_superseded"})
                if any(not write.get("in_place") for write in history[position + 1 :]):
                    return SkipDecision(False, f"input_changed:{key}", {"reason": "reproduced"})
            elif observed["sha256"] != rec.get("sha256"):
                return SkipDecision(False, f"input_changed:{key}", {"recorded": rec.get("sha256"), "observed": observed["sha256"]})

        if entry.get("values_sha256") != canonical_json_digest(dict(step.values)):
            return SkipDecision(False, "values_changed")

        if adopted:
            self.save()
        return SkipDecision(True, "current", {"adopted": adopted})

    def drift(self) -> list[str]:
        """Recorded paths whose on-disk state no longer matches the ledger.

        Cheap whole-deck probe for a stage-level resume short-circuit: a
        succeeded stage receipt may only be reused while no file any deck
        step wrote (history head) or read (external inputs) has moved. It
        answers *whether* the chain must be walked, not which steps re-run --
        the per-step ``should_skip`` decides that.
        """

        drifted: list[str] = []
        for key, entry in self._data["paths"].items():
            entry = entry if isinstance(entry, Mapping) else {}
            history = entry.get("history") if isinstance(entry.get("history"), list) else []
            if not history:
                continue
            observed = self._observe(self._resolve(key), pattern=entry.get("pattern"))
            if not observed["exists"] or observed["sha256"] != history[-1].get("sha256"):
                drifted.append(key)
        for entry in self._data["steps"].values():
            inputs = entry.get("inputs") if isinstance(entry, Mapping) else None
            for key, rec in (inputs or {}).items():
                if key in self._data["paths"] or key in drifted:
                    continue
                rec = rec if isinstance(rec, Mapping) else {}
                observed = self._observe(self._resolve(key), pattern=rec.get("pattern"), recorded=rec)
                if bool(observed["exists"]) != bool(rec.get("exists")) or (
                    observed["exists"] and observed["sha256"] != rec.get("sha256")
                ):
                    drifted.append(key)
        return sorted(set(drifted))

    # -- reporting ----------------------------------------------------------

    def output_digest(self) -> str:
        heads = {key: self._head_sha(key) for key in sorted(self._data["paths"])}
        return canonical_json_digest(heads)

    def ledger_sha256(self) -> str | None:
        return sha256_file(self.ledger_path) if self.ledger_path.is_file() else None

    def summary(self) -> dict[str, Any]:
        by_status: dict[str, list[str]] = {}
        for step_id, entry in sorted(self._data["steps"].items()):
            status = str(entry.get("last_status") or "unknown") if isinstance(entry, Mapping) else "unknown"
            by_status.setdefault(status, []).append(step_id)
        decisions = {
            step_id: entry.get("last_decision")
            for step_id, entry in sorted(self._data["steps"].items())
            if isinstance(entry, Mapping) and isinstance(entry.get("last_decision"), str)
        }
        return {
            "steps_run": by_status.get("ran", []),
            "steps_reused": by_status.get("reused", []),
            "steps_adopted": by_status.get("adopted", []),
            "path_count": len(self._data["paths"]),
            "decisions": decisions,
        }

    def report_entry(self) -> dict[str, Any]:
        return {
            "ledger": self._key(self.ledger_path),
            "ledger_sha256": self.ledger_sha256(),
            "output_digest": self.output_digest(),
            **self.summary(),
        }


def presentation_output_digest(ledgers: Mapping[str, DeckStepLineage]) -> str:
    """Stage-level digest over every deck's ledger output digest."""

    return canonical_json_digest({deck_id: ledger.output_digest() for deck_id, ledger in sorted(ledgers.items())})


def load_deck_output_digests(workspace: Path, deck_dirs: Iterable[Path]) -> dict[str, str]:
    return {
        Path(deck_dir).name: DeckStepLineage.for_deck_dir(Path(deck_dir), workspace=workspace).output_digest()
        for deck_dir in deck_dirs
    }


def record_presentation_command(
    command: Sequence[str],
    *,
    workspace: Path,
    deck_dir: Path,
    outputs: Iterable[Path],
    snapshot: Mapping[str, Any] | None = None,
    **spec_kwargs: Any,
) -> StepSpec:
    """Record a just-completed argv-driven step (ICLR/ICML per-command port)."""

    ledger = DeckStepLineage.for_deck_dir(Path(deck_dir), workspace=workspace)
    spec = StepSpec.from_command(command, workspace=workspace, outputs=outputs, **spec_kwargs)
    ledger.record(spec, snapshot if snapshot is not None else ledger.snapshot_inputs(spec))
    return spec
