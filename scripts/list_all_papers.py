#!/usr/bin/env python3
"""
List all ranked papers across all NeurIPS 2025 presentation topics.
Deduplicates by title, parses published dates, and prints a sorted table.
"""

from pathlib import Path
import json
import glob
import re
import os
from datetime import datetime

BASE = (str(Path(__file__).resolve().parents[1]) + "/conferences/neurips2025/decks")

MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def parse_date(published_str):
    """
    Parse a published string like 'September 2023', 'July 2024', '2024',
    or messy variants like '2024 (MergeKit), Tutorial December 2025'.
    Returns a (datetime, display_string) tuple.
    Falls back to datetime.max for unparseable strings.
    """
    if not published_str or published_str == "None":
        return datetime(9999, 1, 1), "Unknown"

    s = published_str.strip()

    # Try "Month YYYY" pattern first (take the first one found)
    match = re.search(
        r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})',
        s, re.IGNORECASE
    )
    if match:
        month_name = match.group(1).lower()
        year = int(match.group(2))
        month = MONTH_MAP[month_name]
        display = f"{match.group(1)} {year}"
        return datetime(year, month, 1), display

    # Try bare year "YYYY"
    match = re.search(r'(\d{4})', s)
    if match:
        year = int(match.group(1))
        return datetime(year, 1, 1), str(year)

    # Fallback
    return datetime(9999, 1, 1), s[:20]


def extract_topic_label(topic_str):
    """Convert '01_tools_and_open_source' -> '01 Tools & Open Source'."""
    match = re.match(r'^(\d+)_(.+)$', topic_str)
    if match:
        num = match.group(1)
        name = match.group(2).replace('_', ' ').title()
        return f"{num} {name}"
    return topic_str


def main():
    files = sorted(glob.glob(os.path.join(BASE, "*/ranked_papers.json")))
    if not files:
        print("No ranked_papers.json files found!")
        return

    # Collect all papers, tracking topics per title
    seen = {}        # title_key -> paper dict (first occurrence)
    topics_map = {}  # title_key -> list of topic labels

    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)

        topic_label = extract_topic_label(
            data.get("topic", os.path.basename(os.path.dirname(fpath)))
        )

        for paper in data["papers"]:
            title = paper.get("title", "").strip()
            if not title:
                continue

            # Normalize title for dedup (lowercase, collapse whitespace)
            title_key = re.sub(r'\s+', ' ', title.lower())

            if title_key not in seen:
                seen[title_key] = paper
                topics_map[title_key] = []
            topics_map[title_key].append(topic_label)

    # Parse dates and build sortable list
    rows = []
    for title_key, paper in seen.items():
        dt, date_display = parse_date(str(paper.get("published", "")))
        title = paper.get("title", "")
        citations = paper.get("citations", 0) or 0
        mentions = paper.get("mentions", 0) or 0
        pdf_exists = paper.get("pdf_exists", False)
        topics = topics_map[title_key]

        rows.append({
            "sort_date": dt,
            "date_display": date_display,
            "title": title,
            "citations": citations,
            "mentions": mentions,
            "has_pdf": "Yes" if pdf_exists else "No",
            "topics": ", ".join(topics),
        })

    # Sort by date ascending, then title
    rows.sort(key=lambda r: (r["sort_date"], r["title"].lower()))

    # Print table
    col_date = 18
    col_title = 72
    col_topics = 34
    col_cite = 10
    col_ment = 9
    col_pdf = 7

    header = (
        f"{'Published':<{col_date}} "
        f"{'Title':<{col_title}} "
        f"{'Topic(s)':<{col_topics}} "
        f"{'Citations':>{col_cite}} "
        f"{'Mentions':>{col_ment}} "
        f"{'Has PDF':>{col_pdf}}"
    )
    sep = "-" * len(header)

    print(sep)
    print(header)
    print(sep)

    for row in rows:
        title_trunc = row["title"][:col_title - 2]
        if len(row["title"]) > col_title - 2:
            title_trunc += ".."

        topics_trunc = row["topics"][:col_topics - 2]
        if len(row["topics"]) > col_topics - 2:
            topics_trunc += ".."

        line = (
            f"{row['date_display']:<{col_date}} "
            f"{title_trunc:<{col_title}} "
            f"{topics_trunc:<{col_topics}} "
            f"{row['citations']:>{col_cite},} "
            f"{row['mentions']:>{col_ment}} "
            f"{row['has_pdf']:>{col_pdf}}"
        )
        print(line)

    print(sep)
    print(f"\nTotal unique papers: {len(rows)}")
    print(f"  With PDF: {sum(1 for r in rows if r['has_pdf'] == 'Yes')}")
    print(f"  Without PDF: {sum(1 for r in rows if r['has_pdf'] == 'No')}")
    print(f"  Topics covered: {len(files)}")

    # Date distribution
    from collections import Counter
    year_counts = Counter()
    for r in rows:
        if r["sort_date"].year < 9999:
            year_counts[r["sort_date"].year] += 1
        else:
            year_counts["Unknown"] += 1
    print(f"\n  Papers by year:")
    for yr in sorted(year_counts.keys(), key=lambda x: (isinstance(x, str), x)):
        print(f"    {yr}: {year_counts[yr]}")


if __name__ == "__main__":
    main()
