"""The numeric-claims ledger.

A quantitative figure may only plot a number that a source actually stated. The
ledger is the record of those numbers: one row per claim, each carrying the
source it came from and the span of text that states it, so a chart can be
traced back to a sentence and the artifact-truth validator can check that it
was.

Extraction is deterministic and free: it reads the deck's own bound sources
(the speaker/author/work ledger's contribution notes, plus any abstract text in
the narrative source pack) and matches numbers together with the metric words
around them. No model call, so it can run on every build.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LEDGER_VERSION = "numeric-claims/1"

# A number with a unit, and enough of its sentence to say what it measures.
# The number must start a token: "x0", "GPT-4", "3D" and "COCO-2017" are names,
# not measurements, and a chart built on one of them would be a chart about
# nothing.
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9.\-])"
    r"(?P<value>\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*"
    r"(?P<unit>%|percent|x\b|×|-fold|fold|B\b|M\b|K\b|GB\b|ms\b|FPS\b|fps\b|pp\b|points?\b)?",
)

# Words that make a number a *claim* rather than an incidental figure (a year, a
# section number, a count of authors).
_METRIC_WORDS = (
    "accuracy", "speedup", "speed-up", "latency", "throughput", "recall", "precision",
    "f1", "score", "improvement", "reduction", "reduced", "gain", "drop", "error",
    "faster", "slower", "smaller", "larger", "fewer", "pruned", "retained",
    "compression", "memory", "parameters", "tokens", "frames", "resolution", "map",
    "iou", "psnr", "ssim", "fid", "win rate", "success rate", "coverage",
    "robustness", "steps", "budget",
)

_YEAR_RE = re.compile(r"^(19|20)\d{2}$")


@dataclass(frozen=True)
class NumericClaim:
    """One measured number, with the sentence that states it."""

    claim_id: str
    source_id: str
    span: str
    metric: str
    value: float
    unit: str
    comparator: str = ""
    baseline: str = ""
    condition: str = ""
    confidence: float = 0.0

    @property
    def plottable(self) -> bool:
        """Can this claim fill an axis, or only a sentence?

        A quantitative figure needs a number, a unit, and a name for what was
        measured. Extraction often lands the first two and leaves `metric`
        empty -- the span says something happened, but not what quantity it is
        -- and half the rows on both CVPR decks are like that. Charting one
        means inventing the axis label, which is the exact failure the ledger
        exists to prevent, so a row without a metric grounds narration and not
        a chart.
        """

        return bool(str(self.metric).strip()) and bool(str(self.unit).strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.claim_id,
            "source_id": self.source_id,
            "span": self.span,
            "metric": self.metric,
            "value": self.value,
            "unit": self.unit,
            "comparator": self.comparator,
            "baseline": self.baseline,
            "condition": self.condition,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class NumericLedger:
    """Every numeric claim available to one deck."""

    topic_id: str
    claims: list[NumericClaim] = field(default_factory=list)
    sources_scanned: int = 0
    _by_id: dict[str, NumericClaim] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._by_id = {claim.claim_id: claim for claim in self.claims}

    def get(self, claim_id: str) -> NumericClaim | None:
        return self._by_id.get(claim_id)

    def has(self, claim_id: str) -> bool:
        return claim_id in self._by_id

    def for_source(self, source_id: str) -> list[NumericClaim]:
        return [claim for claim in self.claims if claim.source_id == source_id]

    # A single shared word is enough to OFFER a claim to a prompt -- the author
    # can ignore it. It is not enough to ASSERT that a section has measured
    # claims of its own: at that bar every section of every deck looks grounded,
    # because a handful of claims each share one word with everything.
    STRONG_MATCH_SCORE = 3

    def plottable_search(self, terms: str, *, limit: int = 8,
                         min_score: int = 1) -> list[NumericClaim]:
        """`search`, keeping only claims that could actually fill a chart."""

        return [claim for claim in self.search(terms, limit=limit, min_score=min_score)
                if claim.plottable]

    def search(
        self, terms: str, *, limit: int = 8, min_score: int = 1,
        sources: Collection[str] | None = None,
    ) -> list[NumericClaim]:
        """Claims whose span shares the most distinctive words with ``terms``.

        ``min_score`` is the number of distinctive words that must overlap.
        Leave it at 1 to offer candidates; raise it to ``STRONG_MATCH_SCORE``
        to ask whether the ledger really has something to say about a slide.

        ``sources`` is the slide's OWN cited sources, and a row from anywhere
        else cannot ground it however well the words match. Without it a deck
        about video matched "accuracy", "benchmark" and "video" across every
        paper in the corpus: 22 of 25 strong matches on decks 04 and 05 were
        another work's numbers, and three figures shipped plotting them --
        deck 01's slide 5 attributing a 55% exploration figure to a slide on
        aggregate accuracy, deck 04's slide 20 putting a 2.5B parameter count
        on a slide about manipulation quality, deck 05's slide 6 borrowing
        SPECT's latency. Compared at document granularity, so another segment
        of the same talk still counts.
        """

        wanted = _tokens(terms)
        if not wanted:
            return []
        claims = self.claims
        if sources is not None:
            allowed = {_source_key(value) for value in sources}
            claims = [claim for claim in claims if _source_key(claim.source_id) in allowed]
        scored = [
            (len(wanted & _tokens(claim.span)) + len(wanted & _tokens(claim.metric)), claim)
            for claim in claims
        ]
        hits = [(score, claim) for score, claim in scored if score >= max(1, min_score)]
        hits.sort(key=lambda pair: (-pair[0], pair[1].claim_id))
        return [claim for _, claim in hits[:limit]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ledger_version": LEDGER_VERSION,
            "topic_id": self.topic_id,
            "sources_scanned": self.sources_scanned,
            "claim_count": len(self.claims),
            "claims": [claim.to_dict() for claim in self.claims],
        }

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path


def _source_key(source_id: object) -> str:
    """A source id without its segment, so any segment of one talk is one source."""

    return str(source_id or "").rsplit(":segment:", 1)[0]


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9][a-z0-9\-]{2,}", str(text).lower())}


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", str(text or "").strip())
    return [part.strip() for part in parts if part.strip()]


def _metric_for(sentence: str) -> str:
    lowered = sentence.lower()
    for word in _METRIC_WORDS:
        if word.lower() in lowered:
            return word
    return ""


def _normalise_unit(raw: str) -> str:
    unit = (raw or "").strip()
    if unit.lower() in {"percent", "%"}:
        return "%"
    if unit in {"×", "x"}:
        return "x"
    if unit.lower() in {"-fold", "fold"}:
        return "x"
    return unit


def claims_in_text(text: str, *, source_id: str, prefix: str, start_index: int = 0) -> list[NumericClaim]:
    """Every numeric claim stated in ``text``.

    A number counts as a claim when it carries a unit, or when it is a decimal
    in a sentence that names a metric. A bare integer with no unit does not:
    "three benchmarks" and "Figure 2" are not measurements, and a chart drawn
    from one would be a chart about nothing. Identical claims from the same
    sentence are recorded once.
    """

    claims: list[NumericClaim] = []
    seen: set[tuple[float, str, str]] = set()
    index = start_index
    for sentence in _sentences(text):
        metric = _metric_for(sentence)
        for match in _NUMBER_RE.finditer(sentence):
            raw_value = match.group("value")
            unit = _normalise_unit(match.group("unit") or "")
            digits = raw_value.replace(",", "")
            if _YEAR_RE.match(digits) and not unit:
                continue
            if not unit and "." not in digits:
                continue  # a bare integer is not a measurement
            if not unit and not metric:
                continue
            try:
                value = float(digits)
            except ValueError:  # pragma: no cover - the regex only matches numbers
                continue
            if value == 0.0:
                continue
            key = (value, unit, sentence)
            if key in seen:
                continue
            seen.add(key)
            index += 1
            confidence = 0.9 if unit and metric else 0.7 if unit else 0.5
            claims.append(
                NumericClaim(
                    claim_id=f"{prefix}-{index:04d}",
                    source_id=source_id,
                    span=sentence.strip(),
                    metric=metric,
                    value=value,
                    unit=unit,
                    confidence=confidence,
                )
            )
    return claims


def build_ledger(
    *,
    topic_id: str,
    source_ledger: dict[str, Any] | None = None,
    extra_texts: dict[str, str] | None = None,
) -> NumericLedger:
    """Extract the numeric claims available to one deck.

    ``source_ledger`` is the deck's ``speaker_author_work_ledger.json`` payload;
    its ``contribution_note`` per entry already carries most of the measured
    claims a deck cites. ``extra_texts`` maps a source id to any further text
    (an abstract, a full-read summary) to scan.
    """

    prefix = "nc-" + hashlib.sha256(topic_id.encode("utf-8")).hexdigest()[:4]
    collected: list[NumericClaim] = []
    scanned = 0
    for entry in (source_ledger or {}).get("entries") or []:
        if not isinstance(entry, dict):
            continue
        source_id = str(entry.get("source_id") or entry.get("resolved_title") or "")
        note = str(entry.get("contribution_note") or "")
        if not source_id or not note.strip():
            continue
        scanned += 1
        collected.extend(claims_in_text(note, source_id=source_id, prefix=prefix))
    for source_id, text in sorted((extra_texts or {}).items()):
        scanned += 1
        collected.extend(claims_in_text(text, source_id=source_id, prefix=prefix))

    # One talk is often split across several ledger entries that share a
    # contribution note, so the same measured claim arrives many times. Keep the
    # first occurrence and number the survivors, so ids stay dense and stable.
    claims: list[NumericClaim] = []
    seen: set[tuple[float, str, str]] = set()
    for claim in collected:
        key = (claim.value, claim.unit, claim.span)
        if key in seen:
            continue
        seen.add(key)
        claims.append(
            NumericClaim(
                claim_id=f"{prefix}-{len(claims) + 1:04d}",
                source_id=claim.source_id,
                span=claim.span,
                metric=claim.metric,
                value=claim.value,
                unit=claim.unit,
                confidence=claim.confidence,
            )
        )
    return NumericLedger(topic_id=topic_id, claims=claims, sources_scanned=scanned)


def load_ledger(path: Path) -> NumericLedger:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    claims = [
        NumericClaim(
            claim_id=str(row.get("id") or ""),
            source_id=str(row.get("source_id") or ""),
            span=str(row.get("span") or ""),
            metric=str(row.get("metric") or ""),
            value=float(row.get("value") or 0.0),
            unit=str(row.get("unit") or ""),
            comparator=str(row.get("comparator") or ""),
            baseline=str(row.get("baseline") or ""),
            condition=str(row.get("condition") or ""),
            confidence=float(row.get("confidence") or 0.0),
        )
        for row in payload.get("claims") or []
    ]
    return NumericLedger(
        topic_id=str(payload.get("topic_id") or ""),
        claims=claims,
        sources_scanned=int(payload.get("sources_scanned") or 0),
    )


def verify_spec_grounding(spec_payload: dict[str, Any], ledger: NumericLedger) -> list[str]:
    """Rows a spec cites that the ledger does not hold."""

    grounding = spec_payload.get("grounding") or {}
    rows = grounding.get("ledger_rows") or []
    return [str(row) for row in rows if not ledger.has(str(row))]
