"""Is a figure component named by the slide it illustrates?

The figure brief a blueprint carries for each slide is written by the
editorial model in the same call as the slide, with the whole evidence
digest in context and an instruction to name several components. With two
to four sources per slide, the model fills the list with baselines,
comparators and stages the sources never mention ("Hand-designed harness",
"Fixed patch grid", "Raw 3D conformers" on ICML 2026 decks 01/02). Downstream,
the spec author is told to draw the brief's components, so the invention
reaches the slide, and the critic cannot repair it in three rounds because the
brief keeps asking for it.

This module is the one grounding rule both ends apply: the blueprint drops an
ungrounded component from the brief it compiles, and the spec author drops it
from the brief it is shown. A component is grounded when the slide's own text
or its sources name it: the exact phrase appears, or (for a phrase of three or
more content words) most of its content words do. Short phrases need the exact
phrase because two generic words ("Policy update", "Physics simulation") match
almost any slide on the topic word by word.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

STOPWORDS = frozenset(
    "the a an of and or to in on for by with as vs versus from into at is are its their than "
    "over under per not no yes one two three each all any new via that this these those it be".split()
)
LONG_PHRASE_WORDS = 3
LONG_PHRASE_SHARE = 2 / 3

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[\-\+\.][a-z0-9]+)*")


def normalize(text: Any) -> str:
    """Lowercase, hyphens and punctuation to spaces, single-spaced."""

    return " ".join(re.sub(r"[^a-z0-9\+\.]+", " ", str(text or "").lower()).split())


def stem(word: str) -> str:
    """Crude, consistent suffix strip: 'hacked', 'hacking' and 'hack' meet;
    'candidates' and 'candidate' meet; 'trajectories' meets 'trajectory'."""

    if len(word) > 5 and word.endswith("ies"):
        word = word[:-3] + "y"
    elif len(word) > 6 and word.endswith("ings"):
        word = word[:-4]
    elif len(word) > 5 and word.endswith("ing"):
        word = word[:-3]
    elif len(word) > 5 and word.endswith("ed"):
        word = word[:-2]
    elif len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    if len(word) > 4 and word.endswith("e"):
        word = word[:-1]
    return word


def content_words(text: Any) -> set[str]:
    """Stemmed content words; hyphenated compounds split ("reward-hacked" -> reward, hack)."""

    return {
        stem(w) for w in _TOKEN_RE.findall(normalize(text))
        if len(w) >= 3 and w not in STOPWORDS
    }


# Words that describe a component's place in a figure, not what it is:
# "Generation column", "Validity axis: ...", "Oversight signal A".
STRUCTURAL_WORDS = frozenset(
    stem(w) for w in
    "column row axis gate step stage check score log signal node box panel item entry "
    "input output result candidate example".split()
)
_NAME_TOKEN_RE = re.compile(
    r"\b(?:[A-Z][a-z]+[A-Z][A-Za-z0-9\-]*|[A-Z]{3,}[A-Za-z0-9\-]*|[A-Za-z]{2,}[\-]?\d+[A-Za-z0-9\-]*)\b"
    r"|\(([A-Z][A-Za-z0-9\-]+)[^)]*\)"  # a parenthesised name: "(Walrus)", "(MUSA-PINN)"
)


def name_tokens(component: Any) -> set[str]:
    """CamelCase, acronyms and digit-bearing names: CatFlow, MUSA-PINN, RAGEN-2."""

    found: set[str] = set()
    for match in _NAME_TOKEN_RE.finditer(str(component or "")):
        token = match.group(1) or match.group(0)
        if normalize(token):
            found.add(normalize(token))
    return found


def grounding_text(slide: Mapping[str, Any]) -> str:
    """Everything a slide says and cites, as one normalized string.

    Reads both the blueprint slide shape (``narration_beats``,
    ``speaker_author_work_mentions``, ``source_plan[].supports``) and the deck
    contract shape (``narration_hooks``, ``named_paper_talk_speaker_author_mentions``,
    ``approved_sources``, ``approved_narration``).
    """

    parts: list[str] = []
    for key in ("title", "visible_text", "claims", "narration_beat", "narration_beats", "narration_hooks", "approved_narration"):
        value = slide.get(key)
        if isinstance(value, list):
            parts.extend(str(v) for v in value)
        elif value:
            parts.append(str(value))
    for key in ("speaker_author_work_mentions", "named_paper_talk_speaker_author_mentions", "approved_sources"):
        for mention in slide.get(key) or []:
            if not isinstance(mention, Mapping):
                continue
            parts.append(str(mention.get("title") or ""))
            parts.extend(str(w) for w in mention.get("works") or [])
            parts.append(str(mention.get("contribution_note") or ""))
            parts.append(str(mention.get("supports") or ""))
            parts.extend(str(p) for p in (mention.get("authors") or []) + (mention.get("speakers") or []))
    for plan in slide.get("source_plan") or []:
        if isinstance(plan, Mapping):
            parts.append(str(plan.get("supports") or ""))
    return normalize(" ".join(parts))


STRONG, WEAK, NONE = 2, 1, 0


def grounding_level(component: Any, pool_text: str, pool_words: set[str] | None = None) -> tuple[int, float]:
    """``(level, share)``: STRONG when the exact phrase or one of its name
    tokens (CatFlow, Walrus, MUSA-PINN) appears in the slide; otherwise WEAK
    with the share of its content words present (structural words such as
    "column" and "axis" never count); NONE when no content word is present."""

    text = str(component or "")
    phrase = normalize(text)
    head = normalize(re.sub(r"\(.*?\)", " ", text))
    if pool_words is None:
        pool_words = {stem(w) for w in _TOKEN_RE.findall(pool_text)}
    if (phrase and phrase in pool_text) or (head and head in pool_text):
        return STRONG, 1.0
    names = name_tokens(text)
    if names and any(n in pool_text for n in names):
        return STRONG, 1.0
    words = content_words(text) - STRUCTURAL_WORDS
    if not words:
        return STRONG, 1.0
    share = len(words & pool_words) / len(words)
    return (WEAK if share > 0 else NONE), share


def is_grounded(component: Any, pool_text: str, pool_words: set[str] | None = None, *, strict: bool = False) -> bool:
    """True when the slide's text or sources name this component: by phrase
    or name token, by every content word when ``strict`` (an item of a
    comparison), or by most content words of a phrase of three or more."""

    level, share = grounding_level(component, pool_text, pool_words)
    if level == STRONG:
        return True
    if strict:
        return share >= 1.0
    return len(content_words(component) - STRUCTURAL_WORDS) >= LONG_PHRASE_WORDS and share >= LONG_PHRASE_SHARE


COMPARISON_DEPICTIONS = frozenset({"comparison_matrix", "tradeoff_map", "evidence_chart", "feature_matrix",
                                   "quadrant_map", "scatter_frontier", "contrast_pair", "ranked_list",
                                   "bar_compare", "landscape_regions"})
# Words that mark a foil the brief invented to have something to compare
# against: "Hand-designed harness", "Traditional ML predictor (baseline)",
# "Cutoff-radius baseline GNN", "Single-agent baseline".
COMPARATOR_MARKERS = frozenset(
    "baseline traditional conventional vanilla naive standard manual hand designed "
    "design authored author crafted craft handcrafted plain generic typical legacy".split()
)


def _is_comparator_foil(component: str) -> bool:
    tokens = set(normalize(component).split())
    tokens |= {stem(t) for t in tokens}
    return bool(tokens & COMPARATOR_MARKERS) or " vs " in f" {normalize(component)} "


def looks_named(component: str) -> bool:
    """A system, method, benchmark, work or person rather than a generic stage
    label ("Observation", "Final report"): a name token, a capitalised word
    after the first, or a parenthesised name."""

    text = str(component or "").strip()
    rest = text.split(" ", 1)[1] if " " in text else ""
    return bool(name_tokens(text) or re.search(r"\b[A-Z][a-z]+", rest) or re.search(r"\([A-Z]", text))


def _twin_of_grounded(component: str, grounded: Iterable[str]) -> bool:
    """An ungrounded component that shares most of its words with a grounded
    one is its invented foil ("Fixed patch grid" beside "Jittered patch grid")."""

    words = content_words(component) - STRUCTURAL_WORDS
    if len(words) < 2:
        return False
    for other in grounded:
        shared = words & content_words(other)
        if len(shared) >= 2 and len(shared) / len(words) >= 0.5:
            return True
    return False


def filter_components(
    components: Iterable[Any], slide: Mapping[str, Any], *, depiction: str = ""
) -> tuple[list[str], list[str]]:
    """``(kept, dropped)`` for a brief's components against its slide.

    Items of a comparison (rows, points, bars) are the compared entities and
    must each be named by the slide or its sources. In a mechanism or
    architecture figure a generic stage label may pass unnamed; what is dropped
    there is a named entity the slide never mentions, a comparator foil
    (``COMPARATOR_MARKERS``), or a twin of a named component whose only
    support is the words it shares with that twin ("Fixed patch grid" beside
    "Jittered patch grid (Walrus)").
    """

    pool_text = grounding_text(slide)
    pool_words = {stem(w) for w in _TOKEN_RE.findall(pool_text)}
    strict = str(depiction or "").strip().lower() in COMPARISON_DEPICTIONS
    items = [str(c or "").strip() for c in components if str(c or "").strip()]
    levels = {c: grounding_level(c, pool_text, pool_words) for c in items}
    # Anchors for the twin check: named outright, or every content word present.
    strong = [c for c in items if levels[c][0] == STRONG or levels[c][1] >= 1.0]
    kept: list[str] = []
    dropped: list[str] = []
    for component in items:
        level, share = levels[component]
        if _is_comparator_foil(component):
            dropped.append(component)
        elif level == STRONG:
            kept.append(component)
        elif _twin_of_grounded(component, [c for c in strong if c != component]):
            dropped.append(component)
        elif strict:
            (kept if share >= 1.0 else dropped).append(component)
        elif level == WEAK and len(content_words(component) - STRUCTURAL_WORDS) >= LONG_PHRASE_WORDS and share >= LONG_PHRASE_SHARE:
            kept.append(component)
        elif looks_named(component):
            dropped.append(component)
        else:
            kept.append(component)
    return kept, dropped


def filter_edges(edges: Iterable[Any], kept: Iterable[str]) -> list[str]:
    """Keep only ``A -> B`` edges whose both ends are kept components."""

    keep = {normalize(c) for c in kept}
    out: list[str] = []
    for edge in edges:
        text = str(edge or "")
        if "->" not in text:
            continue
        left, right = (normalize(side) for side in text.split("->", 1))
        if left in keep and right in keep:
            out.append(text.strip())
    return out


def component_sources(label: str, sources: list[dict]) -> set[str]:
    """Attribute named components using unique title/short-name aliases."""
    aliases: dict[str, set[str]] = {}
    for row in sources:
        source = str(row.get('label') or row.get('source_id') or '')
        title = str(row.get('title') or row.get('resolved_title') or '')
        names = [source, title, title.split(':', 1)[0], str(row.get('short_name') or ''),
                 str(row.get('short_title') or '')]
        names.extend(name_tokens(title))
        for name in names:
            alias = normalize(name)
            if len(alias) >= 3 and alias not in STOPWORDS:
                aliases.setdefault(alias, set()).add(source)
    text = ' ' + normalize(label) + ' '
    return {next(iter(ids)) for alias, ids in aliases.items()
            if len(ids) == 1 and ' ' + alias + ' ' in text}
