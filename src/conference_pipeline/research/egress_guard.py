"""Loopback-only network guard for local research runs (conference-neutral).

Wraps a block of code so every outbound socket connection and every name
lookup must stay on this machine (``127.0.0.0/8``, ``::1``, ``localhost``,
Unix sockets). Anything else raises ``EgressBlockedError`` at the call site
and is recorded, so a library that silently falls back to a web service (a
search API, a hosted scorer, telemetry) fails the run instead of sending data
out. Used around GPT-Researcher runs, whose default retriever is a web search.
"""

from __future__ import annotations

import contextlib
import ipaddress
import socket
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

LOCAL_HOSTNAMES = frozenset({"localhost", "localhost.localdomain", "ip6-localhost"})


class EgressBlockedError(ConnectionError):
    """An outbound connection or lookup left the machine inside a guarded block."""


@dataclass
class EgressReport:
    """What the guard saw: loopback connections allowed, anything else blocked."""

    allowed: int = 0
    blocked: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"loopback_connections": self.allowed, "blocked_destinations": sorted(set(self.blocked))}


def is_local_host(host: Any) -> bool:
    if host is None:
        return False
    text = str(host).strip("[]").split("%", 1)[0]
    if text.lower() in LOCAL_HOSTNAMES:
        return True
    try:
        return ipaddress.ip_address(text).is_loopback
    except ValueError:
        return False


def _destination(address: Any) -> str:
    if isinstance(address, tuple) and address:
        return f"{address[0]}:{address[1]}" if len(address) > 1 else str(address[0])
    return str(address)


_LOCK = threading.RLock()
_DEPTH = 0
_REPORTS: list[EgressReport] = []
_ORIGINALS: dict[str, Any] = {}


def _check(family: int, address: Any) -> None:
    if family == getattr(socket, "AF_UNIX", object()):
        return
    host = address[0] if isinstance(address, tuple) and address else address
    if is_local_host(host):
        for report in list(_REPORTS):
            report.allowed += 1
        return
    destination = _destination(address)
    for report in list(_REPORTS):
        report.blocked.append(destination)
    raise EgressBlockedError(f"egress_blocked:{destination}")


def _guarded_connect(self: socket.socket, address: Any) -> Any:
    _check(self.family, address)
    return _ORIGINALS["connect"](self, address)


def _guarded_connect_ex(self: socket.socket, address: Any) -> Any:
    _check(self.family, address)
    return _ORIGINALS["connect_ex"](self, address)


def _guarded_sendto(self: socket.socket, data: Any, *args: Any) -> Any:
    address = args[-1] if args else None
    if address is not None:
        _check(self.family, address)
    return _ORIGINALS["sendto"](self, data, *args)


def _guarded_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
    if host is not None and not is_local_host(host.decode() if isinstance(host, bytes) else host):
        name = host.decode() if isinstance(host, bytes) else str(host)
        for report in list(_REPORTS):
            report.blocked.append(name)
        raise EgressBlockedError(f"egress_blocked_lookup:{name}")
    return _ORIGINALS["getaddrinfo"](host, *args, **kwargs)


@contextlib.contextmanager
def loopback_only() -> Iterator[EgressReport]:
    """Block every non-loopback connection/lookup, process-wide, for the block.

    Re-entrant; the socket functions are restored when the outermost block
    exits. The yielded report lists what was allowed and blocked.
    """

    global _DEPTH
    report = EgressReport()
    with _LOCK:
        if _DEPTH == 0:
            _ORIGINALS.update(
                connect=socket.socket.connect,
                connect_ex=socket.socket.connect_ex,
                sendto=socket.socket.sendto,
                getaddrinfo=socket.getaddrinfo,
            )
            socket.socket.connect = _guarded_connect  # type: ignore[method-assign]
            socket.socket.connect_ex = _guarded_connect_ex  # type: ignore[method-assign]
            socket.socket.sendto = _guarded_sendto  # type: ignore[method-assign]
            socket.getaddrinfo = _guarded_getaddrinfo  # type: ignore[assignment]
        _DEPTH += 1
        _REPORTS.append(report)
    try:
        yield report
    finally:
        with _LOCK:
            _REPORTS.remove(report)
            _DEPTH -= 1
            if _DEPTH == 0:
                socket.socket.connect = _ORIGINALS["connect"]  # type: ignore[method-assign]
                socket.socket.connect_ex = _ORIGINALS["connect_ex"]  # type: ignore[method-assign]
                socket.socket.sendto = _ORIGINALS["sendto"]  # type: ignore[method-assign]
                socket.getaddrinfo = _ORIGINALS["getaddrinfo"]  # type: ignore[assignment]


__all__ = ["EgressBlockedError", "EgressReport", "is_local_host", "loopback_only"]
