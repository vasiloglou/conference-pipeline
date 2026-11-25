"""Find private markers in a source tree or a staged public tree.

A private marker is text that would identify the operator's machine or accounts if it
were published: a home-directory path, a Google Drive account folder, a cloud-storage
mount, a real email address, or code that reads secrets from a shell profile. The rules
here are generic and hold no one's identity. An operator's own literals (usernames,
email addresses) are supplied at run time through ``CONFERENCE_PIPELINE_PRIVATE_MARKERS``
(comma-separated, normally set in the local ``.env``), so the public code never carries
them.

Findings carry a redacted excerpt: every match on the line is replaced by
``<redacted:RULE>``, so a report or record never repeats the marker it found.

Used by ``scripts/audit_private_markers.py``; the policy for public bundles is
``docs/operations/public-release-scrub-policy.md``.
"""

from __future__ import annotations

import fnmatch
import os
import re
import subprocess
import tomllib
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "EXTRA_MARKERS_ENV",
    "redact_text",
    "Allowlist",
    "AllowlistEntry",
    "AllowlistError",
    "MarkerFinding",
    "MarkerRule",
    "ScanReport",
    "builtin_rules",
    "extra_rules_from_env",
    "has_publishing_marker",
    "load_allowlist",
    "PUBLISHING_LITERAL_MARKERS",
    "publishing_literal_rules",
    "publishing_marker_hits",
    "redact_publishing_markers",
    "scan_paths",
    "scan_text",
]

EXTRA_MARKERS_ENV = "CONFERENCE_PIPELINE_PRIVATE_MARKERS"
MAX_FILE_BYTES = 2 * 1024 * 1024
EXCERPT_CHARS = 160
CODE_SUFFIXES = frozenset({".py", ".sh", ".bash", ".zsh"})
# Domains reserved for documentation and tests (RFC 2606 / RFC 6761) and fixture
# spellings this repository uses; an address under one of these identifies no one.
FIXTURE_EMAIL_DOMAINS = frozenset({"example.com", "example.org", "example.net"})
FIXTURE_EMAIL_TLDS = frozenset({"example", "invalid", "test", "localhost"})

_EMAIL = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"


class _LinearEmailPattern:
    """``re.compile(_EMAIL)`` with the same ``finditer`` matches in linear time.

    The plain pattern retries every start in a long run of address characters with no
    "@", which is quadratic (a 3 MB line ran for hours; pr-43-r1). A match's local part
    cannot hold "@", so it ends at the first "@" after its start, and whether it matches
    depends only on the text after that "@". So each "@" is tried once, from the leftmost
    start the plain pattern would try: the earliest local-part character after the last
    match and after the last character that cannot be in a local part. Its matches are
    the plain pattern's own ``re.Match`` objects. (A lookbehind anchor was not equivalent:
    it dropped an address that starts where the previous match ends; phase-7-r2 F1.)"""

    _LOCAL = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._%+-")

    def __init__(self, pattern: str) -> None:
        self._compiled = re.compile(pattern)
        self.pattern = pattern

    def finditer(self, text: str) -> Iterator[re.Match[str]]:
        pos = 0
        at = text.find("@")
        while at != -1:
            start = at
            while start > pos and text[start - 1] in self._LOCAL:
                start -= 1
            if start < at:
                match = self._compiled.match(text, start)
                if match is not None:
                    yield match
                    pos = match.end()
            at = text.find("@", max(at + 1, pos))


_EMAIL_PATTERN = _LinearEmailPattern(_EMAIL)


class AllowlistError(ValueError):
    """Raised when the allowlist file is missing or malformed."""


@dataclass(frozen=True)
class MarkerRule:
    id: str
    pattern: re.Pattern[str] | _LinearEmailPattern
    code_only: bool = False


@dataclass(frozen=True)
class MarkerFinding:
    path: str
    line: int
    rule: str
    excerpt: str

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, "line": self.line, "rule": self.rule, "excerpt": self.excerpt}


@dataclass(frozen=True)
class AllowlistEntry:
    path: str
    rule: str
    match: str
    reason: str

    def covers(self, finding_path: str, rule: str, line_text: str, start: int, end: int) -> bool:
        """Whether this entry exempts the match at ``line_text[start:end]``.

        The entry's ``match`` text must occur on the line overlapping that match, so a
        generic exemption never hides another marker of the same rule on the same line.
        """
        if self.rule not in ("*", rule):
            return False
        if not fnmatch.fnmatch(finding_path, self.path):
            return False
        if self.match in ("*", ""):
            return True
        position = line_text.find(self.match)
        while position != -1:
            if position < end and position + len(self.match) > start:
                return True
            position = line_text.find(self.match, position + 1)
        return False


@dataclass(frozen=True)
class Allowlist:
    entries: tuple[AllowlistEntry, ...] = ()


@dataclass
class ScanReport:
    findings: list[MarkerFinding] = field(default_factory=list)
    suppressed: int = 0
    files_scanned: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    stale_allowlist: list[AllowlistEntry] = field(default_factory=list)
    # (path, reason) for every file the text scan did not read; not part of to_dict(), so
    # existing reports keep their shape. A gate that must fail closed reads it (pr-43-r1 F1).
    skipped_paths: list[tuple[str, str]] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return dict(sorted(Counter(f.rule for f in self.findings).items()))

    def to_dict(self) -> dict[str, object]:
        return {
            "findings": [f.to_dict() for f in self.findings],
            "counts": self.counts(),
            "suppressed": self.suppressed,
            "files_scanned": self.files_scanned,
            "skipped": dict(sorted(self.skipped.items())),
            "stale_allowlist": [
                {"path": e.path, "rule": e.rule, "reason": e.reason} for e in self.stale_allowlist
            ],
        }


def builtin_rules() -> tuple[MarkerRule, ...]:
    """The generic rules: no person, account or machine is named in them."""
    return (
        # Any account name of two or more characters (underscores and non-ASCII
        # included); a placeholder such as /Users/<name> is not a name.
        MarkerRule("home_path", re.compile(r"""/(?:Users|home)/[^\s/"'`<>:;,()\[\]{}]{2,}""")),
        MarkerRule("drive_account", re.compile(r"GoogleDrive-" + _EMAIL)),
        MarkerRule("cloud_storage", re.compile(r"Library/CloudStorage/")),
        MarkerRule("email", _EMAIL_PATTERN),
        MarkerRule(
            "shell_profile_secret",
            re.compile(r"""["'/]\.(?:zshrc|bashrc|bash_profile|profile)\b"""),
            code_only=True,
        ),
    )


def extra_rules_from_env(environ: Mapping[str, str] | None = None) -> tuple[MarkerRule, ...]:
    """Literal markers from ``CONFERENCE_PIPELINE_PRIVATE_MARKERS`` (comma-separated)."""
    raw = (os.environ if environ is None else environ).get(EXTRA_MARKERS_ENV, "")
    literals = sorted({part.strip() for part in raw.split(",") if part.strip()}, key=len, reverse=True)
    if not literals:
        return ()
    return (MarkerRule("extra", re.compile("|".join(re.escape(x) for x in literals))),)


def load_allowlist(path: Path | None) -> Allowlist:
    """Load ``[[allow]]`` entries; ``None`` means an empty allowlist."""
    if path is None:
        return Allowlist()
    if not path.is_file():
        raise AllowlistError(f"allowlist not found: {path}")
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise AllowlistError(f"allowlist is unreadable: {path}: {type(exc).__name__}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise AllowlistError(f"allowlist is not valid TOML: {path}: {exc}") from exc
    items = payload.get("allow", [])
    if not isinstance(items, list):
        raise AllowlistError("allowlist key 'allow' must be an array of tables")
    entries = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise AllowlistError(f"allowlist entry {index} is not a table")
        values = {key: item.get(key, "*" if key == "match" else "") for key in ("path", "rule", "match", "reason")}
        wrong = [key for key, value in values.items() if not isinstance(value, str)]
        if wrong:
            raise AllowlistError(f"allowlist entry {index}: {', '.join(wrong)} must be text")
        missing = [key for key in ("path", "rule", "reason") if not values[key].strip()]
        if missing:
            raise AllowlistError(f"allowlist entry {index} lacks {', '.join(missing)}")
        entries.append(AllowlistEntry(**values))
    return Allowlist(tuple(entries))


def _is_fixture_email(address: str) -> bool:
    domain = address.rsplit("@", 1)[-1].lower()
    return domain in FIXTURE_EMAIL_DOMAINS or domain.rsplit(".", 1)[-1] in FIXTURE_EMAIL_TLDS


def _names_someone(home_path: str) -> bool:
    """``/Users/...`` in prose is a placeholder: a name needs a letter, digit or underscore."""
    name = home_path.split("/", 2)[2]
    return any(ch.isalnum() or ch == "_" for ch in name)


def _merge(spans: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """Overlapping or touching spans become one, labelled by the first rule."""
    merged: list[tuple[int, int, str]] = []
    for start, end, rule in sorted(spans):
        if merged and start <= merged[-1][1]:
            last_start, last_end, last_rule = merged[-1]
            merged[-1] = (last_start, max(last_end, end), last_rule)
        else:
            merged.append((start, end, rule))
    return merged


def _redact(line: str, spans: list[tuple[int, int, str]], *, limit: int | None = EXCERPT_CHARS) -> str:
    out, cursor = [], 0
    for start, end, rule in _merge(spans):
        out.append(line[cursor:start])
        out.append(f"<redacted:{rule}>")
        cursor = end
    out.append(line[cursor:])
    text = "".join(out).strip()
    if limit is None or len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


_TOKEN_EDGE = re.compile(r"""[\s"'`<>()\[\]{}]""")
_WEB_URL_START = re.compile(r"https?://[^/\s]+")


def _inside_web_url(line: str, start: int) -> bool:
    """Whether ``start`` falls in the path of an http(s) URL: the token holding it (cut at
    whitespace, quotes and brackets) begins with ``http(s)://host``. A /home/ or /Users/
    segment there is a web path, not a local home folder (6b, NeurIPS 2025 deck 19)."""
    edges = [m.end() for m in _TOKEN_EDGE.finditer(line, 0, start)]
    token = line[edges[-1] if edges else 0:start]
    return bool(_WEB_URL_START.match(token))


def _match_spans(line: str, rules: Sequence[MarkerRule], is_code: bool) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for rule in rules:
        if rule.code_only and not is_code:
            continue
        for match in rule.pattern.finditer(line):
            if rule.id == "email" and _is_fixture_email(match.group(0)):
                continue
            if rule.id == "home_path" and not _names_someone(match.group(0)):
                continue
            if rule.id == "home_path" and _inside_web_url(line, match.start()):
                continue
            spans.append((match.start(), match.end(), rule.id))
    accounts = [(a, b) for a, b, rule in spans if rule == "drive_account"]
    # An address inside a Drive account folder is that finding, not a second one.
    return [
        (a, b, rule)
        for a, b, rule in spans
        if not (rule == "email" and any(x <= a and b <= y for x, y in accounts))
    ]


def redact_text(text: str, rules: Sequence[MarkerRule]) -> str:
    """``text`` with every marker replaced by ``<redacted:RULE>`` (for paths and messages)."""
    return "\n".join(
        _redact(line, _match_spans(line, rules, True), limit=None) if line else line
        for line in text.split("\n")
    )


# --- the publishing gates' one definition (v6.0-Public-Release Phase 3, D4) -------------
#
# Before Phase 3 each publishing gate kept its own tuple of literal tokens. They now share
# this one: the union of those tokens and the generic rules above, plus the operator's
# own literals, so nothing that blocked before stops blocking (D4 (a)).

PUBLISHING_LITERAL_MARKERS: tuple[str, ...] = (
    "file://",
    "/Users/",
    "/Volumes/",
    "Google Drive",
    "drive.google.com",
    ".env",
)
ENV_FILE_MARKER = ".env"
# ".env" as a file name: the one definition every scan uses, whatever the file type (text,
# PDF, zip member, UTF-16). The token is a file name unless a letter or digit continues it:
# ".env", "prod.env", ".env.local", "(.env)" match; "changes.environmental" does not. Before,
# text used a substring rule, PDFs a narrower one and the hub and release check skipped .env
# findings in PDFs and zips; each of those seams leaked in review (6b r1 F1 to pr-49-r6 F3).
_ENV_FILE_NAME = re.escape(ENV_FILE_MARKER) + r"(?![A-Za-z0-9])"
ENV_FILE_NAME = re.compile(_ENV_FILE_NAME)


def publishing_literal_rules() -> tuple[MarkerRule, ...]:
    """The literal publishing tokens as rules, one per token, named by the token itself.

    ``release check`` adds these to its scan so that it refuses what the site build refuses
    (v6.0 Phase 6, D13): before, a ``drive.google.com`` link passed check and failed the site.
    """
    return tuple(
        MarkerRule(token, ENV_FILE_NAME if token == ENV_FILE_MARKER else re.compile(re.escape(token)))
        for token in PUBLISHING_LITERAL_MARKERS
    )


def _literal_pattern(token: str) -> re.Pattern[str]:
    return ENV_FILE_NAME if token == ENV_FILE_MARKER else re.compile(re.escape(token))


def _publishing_rules(rules: Sequence[MarkerRule] | None) -> Sequence[MarkerRule]:
    return builtin_rules() + extra_rules_from_env() if rules is None else rules


def publishing_marker_hits(
    text: str,
    *,
    allow_env_mention: bool = False,
    rules: Sequence[MarkerRule] | None = None,
) -> list[str]:
    """Which markers ``text`` holds: legacy tokens as themselves, then rule ids, sorted.

    Never returns the matched text. ``allow_env_mention`` exempts the ``.env`` token, for
    the two gates whose output documents a ``.env`` file (BYOK tools, self-host kit).
    Published text is data, so code-only rules do not apply.
    """
    literal = [
        token
        for token in PUBLISHING_LITERAL_MARKERS
        if _literal_pattern(token).search(text) and not (allow_env_mention and token == ENV_FILE_MARKER)
    ]
    # Occurrences of the tokens reported above. An exempted .env mention is not one, so a
    # rule match around it (a home path ending in /.env) is still reported.
    token_spans = [
        (match.start(), match.end())
        for token in literal
        for match in _literal_pattern(token).finditer(text)
    ]
    # One occurrence is one hit: a rule match that overlaps a legacy token occurrence
    # (``/Users/<name>`` is both the token and ``home_path``) is reported as the token.
    ids = sorted(
        {
            rule
            for start, end, rule in _match_spans(text, _publishing_rules(rules), False)
            if not any(start < t_end and t_start < end for t_start, t_end in token_spans)
        }
    )
    return literal + [rule for rule in ids if rule not in literal]


def has_publishing_marker(
    text: str,
    *,
    allow_env_mention: bool = False,
    rules: Sequence[MarkerRule] | None = None,
) -> bool:
    return bool(publishing_marker_hits(text, allow_env_mention=allow_env_mention, rules=rules))


# A published file's bytes, text or binary (v6.0 Phase 4, reviews phase-4-r1 F4 and r2 F2).
# A file that decodes as UTF-8 is text and meets the full rule set, whatever its suffix. A
# file that does not is binary and is scanned for the literal tokens only, since the rule
# set false-matches emails in compressed data. A bare ".env" is four bytes and collides in
# random data, so in opaque bytes (binary or compressed) it counts after a path separator,
# where a path names the file; anywhere a file has text, the one file-name rule applies.
_BINARY_TOKENS = tuple(t.encode() for t in PUBLISHING_LITERAL_MARKERS if t != ENV_FILE_MARKER)
_BINARY_ENV = re.compile(rb"[/\\]" + _ENV_FILE_NAME.encode())
# UTF-16 text in any bytes, with no guess at whether the bytes are UTF-16 (pr-49-r7 F1,
# r8 F1, F3). Every rule but ".env" reads four decoded views (little- and big-endian, at
# either alignment), so a marker of any script is read whole. ".env" is read on the bytes
# themselves, in the one file-name rule's UTF-16 form: no ASCII letter or digit as the next
# character. A view read at the wrong alignment pairs a string's last character with the
# byte after it, and a decoded view would take that for the end of a word (r8 F5); the
# byte form reads the character's own byte, whichever byte stands beside it.
_ENV_UTF16 = (
    re.compile(rb"\.\x00e\x00n\x00v\x00(?![A-Za-z0-9])"),
    re.compile(rb"\x00\.\x00e\x00n\x00v(?!.[A-Za-z0-9])", re.S),
)


def _utf16_views(data: bytes) -> list[str]:
    """``data`` decoded as UTF-16, little- and big-endian, from its first and its second
    byte; a final odd byte is read with a NUL beside it."""
    views = []
    for offset in (0, 1):
        chunk = data[offset:]
        if len(chunk) % 2:
            chunk += b"\0"
        views.extend(chunk.decode(enc, errors="replace") for enc in ("utf-16-le", "utf-16-be"))
    return views


def _utf16_marker(data: bytes, rules: Sequence[MarkerRule] | None) -> bool:
    """Whether ``data`` holds a marker as UTF-16 text; bytes with no NUL hold none."""
    if b"\0" not in data:
        return False
    if any(pattern.search(data) for pattern in _ENV_UTF16):
        return True
    return any(has_publishing_marker(view, allow_env_mention=True, rules=rules) for view in _utf16_views(data))


_ZIP_MAGIC = b"PK\x03\x04"
_ZIP_MEMBER_LIMIT = 64 * 1024 * 1024  # a member larger than this cannot be shown clean
_PDF_MAGIC = b"%PDF-"


def _zip_marker(data: bytes, rules: Sequence[MarkerRule] | None) -> bool:
    """Markers inside a zip container (.pptx, .docx, .xlsx, .zip): compression hides them
    from a byte scan (6b, NeurIPS 2025 PowerPoint decks). Each member is scanned like a file;
    an unreadable archive is treated as holding a marker, so it fails closed."""
    import io
    import zipfile

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            # Member names and the archive comment are text: every rule meets them (pr-49-r7:
            # the raw text scan read them before containers were left to this check).
            names = [info.filename for info in archive.infolist()]
            # Each member's own comment too (pr-49-r8 F4).
            names.extend(info.comment.decode("utf-8", errors="replace") for info in archive.infolist())
            names.append(archive.comment.decode("utf-8", errors="replace"))
            if has_publishing_marker("\n".join(names), rules=rules):
                return True
            for info in archive.infolist():
                if info.is_dir():
                    continue
                if info.file_size > _ZIP_MEMBER_LIMIT:
                    # Too large to scan whole: it fails closed (6b review r1 F2: a scan of
                    # its first part, tokens only, passed a path after the limit).
                    return True
                if has_publishing_marker_bytes(archive.read(info), rules=rules):
                    return True
    except (zipfile.BadZipFile, OSError, RuntimeError, NotImplementedError, EOFError):
        return True
    return False


# Stream filters: an image's encoded bytes hold no text to read; a filter outside both sets
# cannot be shown clean, so it fails closed.
_IMAGE_FILTERS = frozenset({"/DCTDecode", "/JPXDecode", "/JBIG2Decode", "/CCITTFaxDecode"})
_TEXT_FILTERS = frozenset({"/FlateDecode", "/ASCIIHexDecode", "/ASCII85Decode", "/LZWDecode", "/RunLengthDecode"})
# Any length: a short hex string holds a short marker, and an odd one is read with a final 0
# appended, as a PDF reader does (pr-49-r5 F1).
# PDF whitespace includes NUL, which \s does not (pr-49-r6 F1).
_PDF_WHITESPACE = rb"\x00\t\n\x0b\x0c\r "
_PDF_HEX_STRING = re.compile(rb"<([0-9A-Fa-f" + _PDF_WHITESPACE + rb"]+)>")
# The literal tokens as UTF-16, either byte order: a UTF-16 file or member holds them so
# (6b review r3 F2).
_BINARY_TOKENS_UTF16 = tuple(t.decode().encode(enc) for t in _BINARY_TOKENS for enc in ("utf-16-le", "utf-16-be"))


def _utf16_text(data: bytes) -> str | None:
    """``data`` decoded as UTF-16 when it is UTF-16 text: a byte-order mark, or a NUL in
    most odd (little-endian) or even (big-endian) places of its first 4 KiB."""
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16", errors="replace")
    head = data[:4096]
    if len(head) < 8:
        return None
    half = len(head) // 2
    odd = head[1::2].count(0) / half
    even = head[0::2].count(0) / half
    if odd > 0.4 and even < 0.05:
        return data.decode("utf-16-le", errors="replace")
    if even > 0.4 and odd < 0.05:
        return data.decode("utf-16-be", errors="replace")
    return None


def _binary_marker(data: bytes) -> bool:
    """The literal tokens in bytes with no text to read, as UTF-8 or UTF-16, and ".env" as a
    file name after a path separator."""
    return (any(t in data for t in _BINARY_TOKENS) or any(t in data for t in _BINARY_TOKENS_UTF16)
            or _BINARY_ENV.search(data) is not None)


def _flate_whole(raw: bytes) -> bool:
    """Whether a Flate stream decodes to its end: pypdf returns what it could decode of a
    truncated stream without raising (6b review r2 F3)."""
    import zlib

    decoder = zlib.decompressobj()
    try:
        decoder.decompress(raw)
    except zlib.error:
        return False
    return decoder.eof


def _stream_payload(stream: Any) -> bytes | None:
    """A PDF stream's decoded bytes; ``b""`` for image data, whose encoded bytes the literal
    scan of the file still reads; ``None`` when it cannot be read whole."""
    filters = stream.get("/Filter")
    names = [str(f) for f in filters] if isinstance(filters, list) else ([str(filters)] if filters else [])
    if "/FlateDecode" in names[1:]:
        return None  # a Flate stage after another cannot be checked whole
    if names[:1] == ["/FlateDecode"] and not _flate_whole(stream._data):
        return None
    if any(name in _IMAGE_FILTERS for name in names):
        return b""
    if any(name not in _TEXT_FILTERS for name in names):
        return None
    return stream.get_data()


def _pdf_objects(reader: Any) -> Iterator[Any]:
    """Every object of a PDF: each one its cross-reference table lists, referenced or not,
    and everything inside each, so no structure is left unread because it was not named
    (6b review r3 F1, F4: annotation comments, nested Form streams)."""
    from pypdf.generic import ArrayObject, DictionaryObject, IndirectObject

    stack: list[Any] = [reader.trailer]
    for generation, ids in reader.xref.items():
        stack.extend(IndirectObject(idnum, generation, reader) for idnum in ids)
    stack.extend(IndirectObject(idnum, 0, reader) for idnum in reader.xref_objStm)
    seen: set[tuple[int, int]] = set()
    while stack:
        obj = stack.pop()
        if isinstance(obj, IndirectObject):
            if (obj.idnum, obj.generation) in seen:
                continue
            seen.add((obj.idnum, obj.generation))
            obj = obj.get_object()
        if isinstance(obj, DictionaryObject):  # a stream is a dictionary too
            stack.extend(obj.values())
        elif isinstance(obj, ArrayObject):
            stack.extend(obj)
        if obj is not None:
            yield obj


def _pdf_stream_spans(reader: Any, data: bytes) -> list[tuple[int, int]]:
    """Where each stream's data stands in a PDF's bytes: each object the cross-reference
    table lists is read again from its own offset, as the parser reads it, and its data is
    the bytes just before its endstream keyword. A comment can copy a stream's bytes, never
    its place (pr-49-r7 F2, r8 F2)."""
    from pypdf.generic import StreamObject, read_object

    spans: list[tuple[int, int]] = []
    offsets = sorted({offset for ids in reader.xref.values() for offset in ids.values()})
    for offset in offsets:
        try:
            reader.stream.seek(offset)
            reader.read_object_header(reader.stream)
            obj = read_object(reader.stream, reader)
            after = reader.stream.tell()
        except Exception:  # noqa: BLE001 -- an object that cannot be read again is left as text
            continue
        if not isinstance(obj, StreamObject) or not obj._data:
            continue
        raw = bytes(obj._data)
        stop = data.rfind(b"endstream", offset, after)
        end = stop
        while end >= 0:
            if data[end - len(raw):end] == raw:
                spans.append((end - len(raw), end))
                break
            if end == 0 or data[end - 1:end] not in (b" ", b"\t", b"\r", b"\n", b"\x0c", b"\x00"):
                break
            end -= 1
    return spans


def _pdf_outside_streams(data: bytes, spans: Sequence[tuple[int, int]]) -> bytes:
    """A PDF's bytes with its stream data cut out: object syntax and comments, which no
    reader returns."""
    out, position = [], 0
    for start, end in sorted(spans):
        if start >= position:
            out.append(data[position:start])
            position = end
    out.append(data[position:])
    return b" ".join(out)


def _pdf_marker(data: bytes, rules: Sequence[MarkerRule] | None) -> bool:
    """Markers anywhere in a PDF: its extracted page text, every string object (link
    targets, comments, the information dictionary), and every stream decoded (page and Form
    contents, XMP metadata), which compression hides from a byte scan (6b review r1 F1, r2
    F2, r3 F1 and F4). Text meets every rule, ".env" as a file name only; decoded binary
    data (fonts, image pixels) meets the binary rules. A PDF that is encrypted, or holds a
    stream that cannot be read whole, fails closed (r2 F3)."""
    import io

    try:
        from pypdf import PdfReader
        from pypdf.generic import ByteStringObject, StreamObject, TextStringObject

        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted or len(reader.pages) == 0:  # read leniently, junk parses as an empty document
            return True
        parts = [page.extract_text() or "" for page in reader.pages]
        for obj in _pdf_objects(reader):
            if isinstance(obj, TextStringObject):
                parts.append(str(obj))
            elif isinstance(obj, ByteStringObject):
                parts.append(bytes(obj).decode("utf-8", errors="replace"))
            elif isinstance(obj, StreamObject):
                payload = _stream_payload(obj)
                if payload is None:
                    return True
                # UTF-16 text, however much of the stream is UTF-16 (pr-49-r6 F2, r7 F1, r8 F1).
                if _utf16_marker(payload, rules):
                    return True
                try:
                    parts.append(payload.decode("utf-8"))
                except UnicodeDecodeError:
                    text = _utf16_text(payload)
                    if text is not None:
                        parts.append(text)
                    elif _binary_marker(payload) or _extra_binary_marker(payload, rules):
                        # A decoded binary stream meets the operator's markers too (pr-49-r4 F1).
                        return True
                for match in _PDF_HEX_STRING.finditer(payload):
                    hexed = re.sub(rb"[" + _PDF_WHITESPACE + rb"]", b"", match.group(1))
                    raw = bytes.fromhex((hexed + b"0" * (len(hexed) % 2)).decode())
                    if _binary_marker(raw) or _extra_binary_marker(raw, rules):
                        return True
                    # Its text meets every rule: UTF-16 with a byte-order mark, or UTF-8.
                    text = _utf16_text(raw) if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else None
                    parts.append(text if text is not None else raw.decode("utf-8", errors="replace"))
                    if _utf16_marker(raw, rules):
                        return True
        # Outside its streams a PDF is text, comments included, UTF-8 or UTF-16: every rule
        # meets it there (pr-49-r5 F2, r6 F3, r7 F2, r8 F2 and F3).
        outside = _pdf_outside_streams(data, _pdf_stream_spans(reader, data))
        if _utf16_marker(outside, rules):
            return True
        parts.append(outside.decode("utf-8", errors="replace"))
    except Exception:  # noqa: BLE001 -- any parse failure means the PDF cannot be shown clean
        return True
    text = "\n".join(parts)
    return has_publishing_marker(text, rules=rules)


def _extra_binary_marker(data: bytes, rules: Sequence[MarkerRule] | None) -> bool:
    """The operator's literal markers (``CONFERENCE_PIPELINE_PRIVATE_MARKERS``) in raw bytes.

    Binary bytes otherwise meet the built-in tokens only; the extra rule is literals, so its
    UTF-8 form is searched as bytes without the noise a regex rule would bring (pr-49-r3 F1).
    """
    extra = [r.pattern for r in _publishing_rules(rules) if r.id == "extra" and isinstance(r.pattern, re.Pattern)]
    if not extra:
        return False
    # UTF-8 as bytes, and UTF-16 of either byte order at either alignment (pr-49-r5 F1).
    views = [
        data[offset:].decode(enc, errors="replace") for enc in ("utf-16-le", "utf-16-be") for offset in (0, 1)
    ]
    return any(re.search(p.pattern.encode("utf-8"), data) for p in extra) or any(
        p.search(view) for p in extra for view in views
    )


def has_publishing_marker_bytes(data: bytes, *, rules: Sequence[MarkerRule] | None = None) -> bool:
    """Whether a file's bytes hold a publishing marker (text, UTF-8 or UTF-16: every rule;
    binary: tokens; a zip container: each of its members; a PDF: its text and links too)."""
    if data.startswith(_PDF_MAGIC):
        # A PDF meets the PDF rules whatever its encoding: one that decodes as UTF-8 would
        # otherwise meet the plain-text ".env" rule (6b review r2 F6). Its raw bytes are still
        # checked for the literal tokens.
        return _pdf_marker(data, rules) or _binary_marker(data) or _extra_binary_marker(data, rules)
    try:
        decoded = data.decode("utf-8")
    except UnicodeDecodeError:
        pass
    else:
        # UTF-16 of ASCII text without a byte-order mark is valid UTF-8 too, a NUL beside
        # every character; it is read as UTF-16 below (6b review r3 F2).
        if has_publishing_marker(decoded, rules=rules) or "\0" not in decoded:
            return has_publishing_marker(decoded, rules=rules)
    text = _utf16_text(data)
    if text is not None and has_publishing_marker(text, rules=rules):
        return True
    # UTF-16 text in any bytes, which a guess at the encoding can miss (pr-49-r7 F1, r8 F1).
    if _utf16_marker(data, rules):
        return True
    if data.startswith(_ZIP_MAGIC):
        if _zip_marker(data, rules):
            return True
    return _binary_marker(data) or _extra_binary_marker(data, rules)


def redact_publishing_markers(text: str, *, rules: Sequence[MarkerRule] | None = None) -> str:
    """``text`` with rule matches as ``<redacted:RULE>`` and legacy tokens as ``<private-marker>``."""
    redacted = redact_text(text, _publishing_rules(rules))
    for token in PUBLISHING_LITERAL_MARKERS:
        redacted = redacted.replace(token, "<private-marker>")
    return redacted


def _scan(
    text: str, path: str, rules: Sequence[MarkerRule], allowlist: Allowlist, used: set[int]
) -> tuple[list[MarkerFinding], int]:
    is_code = Path(path).suffix in CODE_SUFFIXES
    shown_path = redact_text(path, rules)
    findings: list[MarkerFinding] = []
    suppressed = 0
    for number, line in enumerate(text.splitlines(), start=1):
        spans = _match_spans(line, rules, is_code)
        if not spans:
            continue
        open_rules: list[str] = []
        for start, end, rule in spans:
            covered = False
            if rule != "extra":
                for index, entry in enumerate(allowlist.entries):
                    if entry.covers(path, rule, line, start, end):
                        used.add(index)
                        covered = True
                        break
            if covered:
                suppressed += 1
            elif rule not in open_rules:
                open_rules.append(rule)
        if open_rules:
            excerpt = _redact(line, spans)
            findings.extend(MarkerFinding(shown_path, number, rule, excerpt) for rule in open_rules)
    return findings, suppressed


def scan_text(
    text: str, path: str, rules: Sequence[MarkerRule], allowlist: Allowlist
) -> list[MarkerFinding]:
    """Findings for one file's text; ``path`` decides code-only rules and the allowlist."""
    findings, _ = _scan(text, path, rules, allowlist, set())
    return findings


def _walk_entries(base: Path) -> list[Path]:
    """Every non-directory entry under ``base``, symlinks to directories included and not
    followed, plus every directory that could not be listed: rglob drops those silently,
    so a file under one would never be seen (phase-7-r4 F2). scan_paths records such a
    directory as not_a_file, and a fail-closed gate reports it."""
    entries: list[Path] = []

    def unreadable(error: OSError) -> None:
        entries.append(Path(error.filename) if error.filename else base)

    for current, dirnames, filenames in os.walk(base, onerror=unreadable, followlinks=False):
        here = Path(current)
        entries.extend(here / name for name in filenames)
        entries.extend(here / name for name in dirnames if (here / name).is_symlink())
    return sorted(set(entries))


def _list_files(root: Path, paths: Sequence[str], tracked_only: bool) -> list[str]:
    if tracked_only:
        result = subprocess.run(
            ["git", "ls-files", "-z", "--", *paths],
            cwd=root,
            capture_output=True,
            check=True,
        )
        return sorted(p for p in result.stdout.decode("utf-8").split("\0") if p)
    # The root and each named path are resolved, as the filesystem reads them: a symlinked
    # root is scanned through, and ".." is taken after symlinks, never before (phase-7-r3
    # F1, F2). Inside, every entry that is not a directory is listed, symlinks (dangling ones
    # too) and FIFOs included, so scan_paths records each one it cannot read instead of
    # never seeing it (phase-7-r2 F3). Directory symlinks inside are listed, not followed.
    files: list[str] = []
    top = root.resolve()
    for item in paths:
        named = root / item
        base = named.resolve()
        if base.is_dir():
            candidates = _walk_entries(base)
        elif base.exists():
            candidates = [base]
        elif named.is_symlink():  # a dangling link named directly: listed, so it is reported
            candidates = [named.parent.resolve() / named.name]
        else:
            candidates = []
        for candidate in candidates:
            rel = candidate.relative_to(top)
            if ".git" in rel.parts:
                continue
            files.append(rel.as_posix())
    return sorted(set(files))


def scan_paths(
    root: Path,
    paths: Sequence[str],
    rules: Sequence[MarkerRule],
    allowlist: Allowlist,
    *,
    tracked_only: bool,
) -> ScanReport:
    """Scan ``paths`` under ``root``: git-tracked files, or every file of a staged tree."""
    report = ScanReport()
    used: set[int] = set()
    skipped: Counter[str] = Counter()
    for rel in _list_files(root, paths, tracked_only):
        file_path = root / rel
        if file_path.is_symlink() or not file_path.is_file():
            skipped["not_a_file"] += 1
            report.skipped_paths.append((rel, "not_a_file"))
            continue
        if file_path.stat().st_size > MAX_FILE_BYTES:
            skipped["oversized"] += 1
            report.skipped_paths.append((rel, "oversized"))
            continue
        data = file_path.read_bytes()
        if b"\0" in data[:8192]:
            skipped["binary"] += 1
            report.skipped_paths.append((rel, "binary"))
            continue
        text = data.decode("utf-8", errors="replace")
        findings, suppressed = _scan(text, rel, rules, allowlist, used)
        report.findings.extend(findings)
        report.suppressed += suppressed
        report.files_scanned += 1
    report.skipped = dict(skipped)
    report.stale_allowlist = [e for i, e in enumerate(allowlist.entries) if i not in used]
    return report
