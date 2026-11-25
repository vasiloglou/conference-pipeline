"""A total host wrapper for the GND command line and its packaged examples.

GND (``get-nick-done``) is installed machine-wide as a ``uv tool``; its packaged
examples (``gnd.examples.literature.*``) run with the tool's own interpreter, found
from the ``gnd`` launcher's shebang. Every function here is total over a missing
binary, a timeout, a non-zero exit, undecodable output and non-JSON output.

An exit code means different things in different GND commands, so the wrapper keeps
the raw integer and classifies it only through :data:`EXIT_MEANINGS`, per command
family. The meanings come from GND's own documentation at the pinned version
(``docs/reference/cli.md`` "Interpret exits in the command's context",
``gnd/witness/types.py`` ``STATUS``, and the literature examples' READMEs).
"""

from __future__ import annotations

import enum
import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The GND version this pipeline is written against: a tool pin, not a conference literal.
GND_VERSION = "0.4.0"

# Sentinels outside the range of real exits: a process killed by signal N returns -N.
NOT_STARTED = -1000
TIMED_OUT = -1001

_EXAMPLE_MODULE = re.compile(r"^literature(\.[a-z_]+)?$")
_VERSION_LINE = re.compile(r"^gnd (\d+\.\d+\.\d+)\s*$")
# The only variables a GND subprocess sees (r1 F6): an allowlist, so no credential of any
# naming scheme (AWS_*, GOOGLE_APPLICATION_CREDENTIALS, *_API_KEY, ...) can reach it.
_ALLOWED_ENV = frozenset({"PATH", "HOME", "TMPDIR", "LANG", "LANGUAGE", "TERM", "USER", "LOGNAME", "SHELL",
                          "PYTHONPATH", "PYTHONHASHSEED", "SOURCE_DATE_EPOCH", "TZ"})
_ALLOWED_PREFIXES = ("LC_",)


class GndUnavailable(RuntimeError):  # noqa: N818 -- name fixed by the approved Phase 1 spec
    """GND is missing, unreadable, or not the pinned version."""


class CommandFamily(enum.StrEnum):
    CHECK = "check"  # gnd check, gnd verify, gnd validate
    ASSURANCE = "assurance"  # literature.assurance_report
    VERDICT = "verdict"  # literature.verdict evaluate
    REPLAY = "replay"  # literature.reading/revision/verdict replay, interop verify
    WITNESS = "witness"  # gnd witness and gnd replay of a certificate
    WHY_NOT = "why-not"  # gnd why-not
    OTHER = "other"  # every other command: 0 = success


EXIT_MEANINGS: Mapping[CommandFamily, Mapping[int, str]] = {
    CommandFamily.CHECK: {0: "pass", 1: "alethic-failure", 2: "vacuous", 3: "input-refused", 4: "unchecked"},
    CommandFamily.ASSURANCE: {0: "all-stages-pass", 1: "stage-failed", 3: "input-refused"},
    CommandFamily.VERDICT: {0: "supported", 2: "support-rejected", 3: "invalid-or-stale"},
    CommandFamily.REPLAY: {0: "replayed", 3: "incomplete"},
    CommandFamily.WITNESS: {
        0: "proved",
        1: "proof-invalid",
        2: "missing-unknown-or-unsupported",
        3: "input-invalid",
        4: "work-limit",
    },
    CommandFamily.WHY_NOT: {0: "answered", 2: "unsupported-rule", 3: "input-invalid", 4: "unavailable-or-limit"},
    CommandFamily.OTHER: {0: "success"},
}


def exit_meaning(family: CommandFamily, exit_code: int) -> str:
    """The documented meaning of ``exit_code`` for ``family``; never guessed."""
    if exit_code == NOT_STARTED:
        return "not-started"
    if exit_code == TIMED_OUT:
        return "timeout"
    if exit_code < 0:
        return f"signal-{-exit_code}"
    return EXIT_MEANINGS[family].get(exit_code, "unmapped")


@dataclass(frozen=True)
class GndResult:
    argv: tuple[str, ...]
    family: CommandFamily
    exit_code: int  # NOT_STARTED / TIMED_OUT sentinels, -N = killed by signal N
    meaning: str
    stdout: str
    stderr: str
    duration_s: float


def _search_path(env: Mapping[str, str] | None) -> str | None:
    return (env if env is not None else os.environ).get("PATH")


def gnd_executable(*, env: Mapping[str, str] | None = None) -> Path:
    """The ``gnd`` launcher on PATH; :class:`GndUnavailable` if there is none."""
    found = shutil.which("gnd", path=_search_path(env))
    if found is None:
        raise GndUnavailable("gnd is not on PATH")
    return Path(found)


def gnd_python(*, env: Mapping[str, str] | None = None) -> Path:
    """The interpreter named by the ``gnd`` launcher's shebang."""
    launcher = gnd_executable(env=env)
    try:
        with launcher.open("rb") as handle:
            first = handle.readline(4096)
    except OSError as exc:
        raise GndUnavailable(f"cannot read the gnd launcher: {exc}") from exc
    line = first.decode("utf-8", errors="replace").strip()
    if not line.startswith("#!"):
        raise GndUnavailable("the gnd launcher has no shebang line")
    parts = line[2:].split()
    if not parts:
        raise GndUnavailable("the gnd launcher's shebang is empty")
    interpreter = Path(parts[-1] if Path(parts[0]).name == "env" else parts[0])
    if Path(parts[0]).name == "env":
        located = shutil.which(parts[-1], path=_search_path(env))
        if located is None:
            raise GndUnavailable(f"the gnd launcher's interpreter {parts[-1]!r} is not on PATH")
        interpreter = Path(located)
    if not interpreter.exists():
        raise GndUnavailable(f"the gnd launcher's interpreter {interpreter} does not exist")
    return interpreter


def scrubbed_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """The allowlisted part of ``env`` (default: the process environment); nothing else passes."""
    source = env if env is not None else os.environ
    return {name: value for name, value in source.items()
            if name in _ALLOWED_ENV or name.startswith(_ALLOWED_PREFIXES)}


def _run(argv: Sequence[str], *, family: CommandFamily, cwd: Path, timeout_s: float,
         env: Mapping[str, str] | None) -> GndResult:
    started = time.monotonic()
    try:
        proc = subprocess.run(
            list(argv), cwd=cwd, env=scrubbed_env(env), stdin=subprocess.DEVNULL,
            capture_output=True, timeout=timeout_s, check=False,
        )
    except OSError as exc:  # missing, not executable, bad format (r1 F8)
        return GndResult(tuple(argv), family, NOT_STARTED, "not-started", "", f"{type(exc).__name__}: {exc}",
                         time.monotonic() - started)
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return GndResult(tuple(argv), family, TIMED_OUT, "timeout", out, err, time.monotonic() - started)
    return GndResult(
        tuple(argv), family, proc.returncode, exit_meaning(family, proc.returncode),
        proc.stdout.decode("utf-8", errors="replace"), proc.stderr.decode("utf-8", errors="replace"),
        time.monotonic() - started,
    )


def run_gnd(args: Sequence[str], *, family: CommandFamily, cwd: Path, timeout_s: float = 600,
            env: Mapping[str, str] | None = None) -> GndResult:
    """Run ``gnd <args>``; a missing binary is ``exit_code`` :data:`NOT_STARTED`, never an exception."""
    try:
        launcher = gnd_executable(env=env)
    except GndUnavailable as exc:
        return GndResult(("gnd", *args), family, NOT_STARTED, "not-started", "", str(exc), 0.0)
    return _run([str(launcher), *args], family=family, cwd=cwd, timeout_s=timeout_s, env=env)


def run_gnd_example(module: str, args: Sequence[str], *, family: CommandFamily, cwd: Path,
                    timeout_s: float = 600, env: Mapping[str, str] | None = None) -> GndResult:
    """Run ``<gnd python> -I -m gnd.examples.<module> <args>``.

    ``module`` must match ``^literature(\\.[a-z_]+)?$``; anything else raises
    :class:`ValueError` before a process starts.
    """
    if not _EXAMPLE_MODULE.match(module):
        raise ValueError(f"not a GND literature example module: {module!r}")
    try:
        interpreter = gnd_python(env=env)
    except GndUnavailable as exc:
        return GndResult(("python", "-m", f"gnd.examples.{module}", *args), family, NOT_STARTED,
                         "not-started", "", str(exc), 0.0)
    return _run([str(interpreter), "-I", "-m", f"gnd.examples.{module}", *args],
                family=family, cwd=cwd, timeout_s=timeout_s, env=env)


def run_gnd_python(args: Sequence[str], *, family: CommandFamily, cwd: Path, timeout_s: float = 600,
                   env: Mapping[str, str] | None = None) -> GndResult:
    """Run ``<gnd python> <args>`` without ``-I``, so ``env`` may set ``PYTHONPATH``.

    Used only for GND's own source-repository helpers (the archived test builders);
    packaged examples go through :func:`run_gnd_example`.
    """
    try:
        interpreter = gnd_python(env=env)
    except GndUnavailable as exc:
        return GndResult(("python", *args), family, NOT_STARTED, "not-started", "", str(exc), 0.0)
    return _run([str(interpreter), *args], family=family, cwd=cwd, timeout_s=timeout_s, env=env)


def gnd_version(*, env: Mapping[str, str] | None = None, timeout_s: float = 60) -> str:
    """The installed GND version, e.g. ``"0.4.0"``; :class:`GndUnavailable` on any failure."""
    result = run_gnd(["--version"], family=CommandFamily.OTHER, cwd=Path.cwd(), timeout_s=timeout_s, env=env)
    if result.exit_code == NOT_STARTED:
        raise GndUnavailable(f"gnd did not start: {result.stderr}")
    if result.exit_code == TIMED_OUT:
        raise GndUnavailable("gnd --version timed out")
    if result.exit_code != 0:
        raise GndUnavailable(f"gnd --version exited {result.exit_code}")
    match = _VERSION_LINE.match(result.stdout.strip())
    if match is None:
        raise GndUnavailable(f"gnd --version printed an unrecognised version line: {result.stdout.strip()[:80]!r}")
    return match.group(1)


def require_gnd(expected: str = GND_VERSION, *, env: Mapping[str, str] | None = None) -> str:
    """Return the installed version if it equals ``expected``; else :class:`GndUnavailable`."""
    found = gnd_version(env=env)
    if found != expected:
        raise GndUnavailable(f"gnd {found} is installed; this pipeline requires gnd {expected}")
    return found


def parse_json_stdout(result: GndResult) -> dict[str, Any] | None:
    """The result's stdout as a JSON object, or ``None`` when it is not one."""
    try:
        value = json.loads(result.stdout)
    except (json.JSONDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None
