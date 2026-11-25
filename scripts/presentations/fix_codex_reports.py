#!/usr/bin/env python3
"""
Fix quality issues in codex-autonomy/paper-influence reports using web search.

Strategy:
1. Parse codex reports for available context (authors, partial titles, venues)
2. Try to match with claude reports first (fastest)
3. For unmatched entries, use Semantic Scholar API to search by author + keywords
4. Generate quality report with recovery statistics

Usage:
    python scripts/presentations/fix_codex_reports.py --topic 01
    python scripts/presentations/fix_codex_reports.py --all
    python scripts/presentations/fix_codex_reports.py --all --dry-run
"""

import argparse
import json
import re
import time
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import requests

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
CODEX_REPORTS = PROJECT_ROOT / "conferences/neurips2025/agent-work/codex/paper-influence/reports"
CLAUDE_REPORTS = PROJECT_ROOT / "conferences/neurips2025/agent-work/claude/paper-influence/reports"
BACKUP_DIR = PROJECT_ROOT / "conferences/neurips2025/agent-work/codex/paper-influence/reports_original_backup"

# Semantic Scholar API (free, no key required for low volume)
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"

# Topic mapping (exact filenames without _influential_papers.md suffix)
TOPIC_NAMES = {
    "01": "01_tools_and_open_source",
    "02": "02_new_architectures_in_genai",
    "03": "03_data_preparation,_cleaning,_and_distillation",
    "04": "04_data_attribution",
    "05": "05_data_markets",
    "06": "06_visual_models",
    "07": "07_tabular_and_relational_models",
    "08": "08_language_models",
    "09": "09_genai_in_science",
    "10": "10_serendipitous_findings",
    "11": "11_ethical_problems",
    "12": "12_commercial_applications",
    "13": "13_industry_insights",
    "14": "14_synthetic_data_generation",
    "15": "15_what's_happening",
    "16": "16_awards",
    "17": "17_llm_scaling_laws_and_efficiency",
    "18": "18_agentic_ai_and_autonomous_systems",
    "19": "19_notable_quotes",
    "20": "20_small_language_models",
}

# Claude report names (may differ slightly)
CLAUDE_TOPIC_NAMES = {
    "01": "01_tools_and_open_source",
    "02": "02_genai_architectures",
    "03": "03_data_preparation",
    "04": "04_data_attribution",
    "05": "05_data_markets",
    "06": "06_visual_models",
    "07": "07_tabular_models",
    "08": "08_language_models",
    "09": "09_genai_in_science",
    "10": "10_serendipitous_findings",
    "11": "11_ethical_problems",
    "12": "12_commercial_applications",
    "13": "13_industry_insights",
    "14": "14_synthetic_data",
    "15": "15_whats_happening_at_neurips",
    "16": "16_neurips_awards",
    "17": "17_llm_scaling_laws_efficiency",
    "18": "18_agentic_ai_autonomous_systems",
    "19": "19_notable_quotes",
    "20": "20_small_language_models",
}


def parse_codex_report(report_path: Path) -> list[dict]:
    """Parse codex influential papers markdown into structured list."""
    content = report_path.read_text()
    papers = []

    # Split by --- separators
    sections = re.split(r'\n---\n', content)

    for section in sections:
        # Find paper header: ## 1. Title or ## 2. Title
        header_match = re.search(r'## (\d+)\. (.+?)(?:\n|$)', section)
        if not header_match:
            continue

        rank = int(header_match.group(1))
        title = header_match.group(2).strip()

        # Extract fields
        authors_match = re.search(r'\*\*Authors:\*\*\s*(.+?)(?:\n|$)', section)
        venue_match = re.search(r'\*\*Venue:\*\*\s*(.+?)(?:\n|$)', section)
        url_match = re.search(r'\*\*URL:\*\*\s*(https?://[^\s\n]+)', section)
        github_match = re.search(r'\*\*GitHub:\*\*\s*(https?://[^\s\n]+)', section)
        published_match = re.search(r'\*\*Published:\*\*\s*(.+?)(?:\n|$)', section)
        citations_match = re.search(r'\*\*Citations:\*\*\s*([^\n]+)', section)

        papers.append({
            "rank": rank,
            "title": title,
            "authors": authors_match.group(1).strip() if authors_match else None,
            "venue": venue_match.group(1).strip() if venue_match else None,
            "url": url_match.group(1).strip() if url_match else None,
            "github_url": github_match.group(1).strip() if github_match else None,
            "published": published_match.group(1).strip() if published_match else None,
            "citations_raw": citations_match.group(1).strip() if citations_match else None,
            "_raw_section": section,
        })

    return papers


def parse_claude_report(report_path: Path) -> list[dict]:
    """Parse claude influential papers markdown into structured list."""
    if not report_path.exists():
        return []

    content = report_path.read_text()
    papers = []

    # Split by --- separators
    sections = re.split(r'\n---\n', content)

    for section in sections:
        # Find paper header
        header_match = re.search(r'## (\d+)\. (.+?)(?:\n|$)', section)
        if not header_match:
            continue

        rank = int(header_match.group(1))
        title = header_match.group(2).strip()

        # Extract fields
        url_match = re.search(r'\*\*URL:\*\*\s*(https?://[^\s\n]+)', section)
        github_match = re.search(r'\*\*GitHub:\*\*\s*(https?://[^\s\n]+)', section)

        papers.append({
            "rank": rank,
            "title": title,
            "url": url_match.group(1).strip() if url_match else None,
            "github_url": github_match.group(1).strip() if github_match else None,
        })

    return papers


def is_valid_arxiv_url(url: str) -> bool:
    """Check if URL is a proper arXiv link (not OpenAlex search)."""
    if not url:
        return False
    return ('arxiv.org/abs/' in url or 'arxiv.org/pdf/' in url) and 'openalex.org' not in url


def is_valid_paper_url(url: str) -> bool:
    """Check if URL is a valid paper URL (arXiv, DOI, or direct)."""
    if not url:
        return False
    if 'openalex.org' in url:
        return False
    valid_domains = ['arxiv.org', 'doi.org', 'openreview.net', 'aclanthology.org',
                     'proceedings.mlr.press', 'papers.nips.cc', 'neurips.cc']
    return any(domain in url for domain in valid_domains)


def title_similarity(t1: str, t2: str) -> float:
    """Calculate title similarity score."""
    if not t1 or not t2:
        return 0.0
    # Normalize titles
    t1_norm = re.sub(r'[^a-z0-9\s]', '', t1.lower())
    t2_norm = re.sub(r'[^a-z0-9\s]', '', t2.lower())
    return SequenceMatcher(None, t1_norm, t2_norm).ratio()


def find_claude_match(codex_paper: dict, claude_papers: list[dict]) -> dict | None:
    """Find best matching paper from claude report."""
    best_match = None
    best_score = 0.0

    # First try exact rank match
    for claude_paper in claude_papers:
        if claude_paper['rank'] == codex_paper['rank']:
            # Check if titles are remotely similar
            score = title_similarity(codex_paper['title'], claude_paper['title'])
            if score >= 0.2:  # Very lenient for rank match
                return claude_paper

    # Fall back to title similarity
    for claude_paper in claude_papers:
        score = title_similarity(codex_paper['title'], claude_paper['title'])
        if score > best_score and score >= 0.3:  # Lower threshold
            best_score = score
            best_match = claude_paper

    return best_match


def search_semantic_scholar(query: str, limit: int = 3, retries: int = 2) -> list[dict]:
    """Search Semantic Scholar for papers matching query with retry."""
    url = f"{SEMANTIC_SCHOLAR_API}/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,authors,year,externalIds,url"
    }

    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 429:
                # Rate limited - wait and retry
                wait_time = 3 * (attempt + 1)
                print(f"      Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            resp.raise_for_status()
            return resp.json().get("data", [])
        except requests.exceptions.HTTPError as e:
            if "429" in str(e):
                time.sleep(3 * (attempt + 1))
                continue
            print(f"      Semantic Scholar error: {e}")
            return []
        except Exception as e:
            print(f"      Semantic Scholar error: {e}")
            return []

    return []


def recover_paper_info(codex_paper: dict, verbose: bool = True) -> dict | None:
    """Attempt to recover correct paper info using available context."""
    search_parts = []

    # Use author names (most reliable)
    authors = codex_paper.get('authors', '')
    if authors and authors not in ('N/A', 'Unknown', ''):
        # Extract first author surname
        first_author = authors.split(',')[0].strip()
        if ' ' in first_author:
            first_author = first_author.split()[-1]
        if len(first_author) > 2:
            search_parts.append(first_author)

    # Use title keywords
    title = codex_paper.get('title', '')
    if title:
        generic = {'neural', 'network', 'learning', 'model', 'system', 'information',
                   'processing', 'systems', 'networks', 'the', 'and', 'for', 'with'}
        keywords = [w for w in title.lower().split()
                    if w not in generic and len(w) > 3 and w.isalpha()]
        search_parts.extend(keywords[:4])

    # Use venue
    venue = codex_paper.get('venue', '')
    if venue and venue not in ('N/A', 'arXiv', ''):
        search_parts.append(venue)

    if len(search_parts) < 2:
        return None

    query = ' '.join(search_parts)
    if verbose:
        print(f"      Searching: {query[:50]}...")

    # Rate limit - Semantic Scholar allows ~100 requests/5 min
    time.sleep(1.0)

    results = search_semantic_scholar(query)
    if results:
        best = results[0]
        arxiv_id = best.get('externalIds', {}).get('ArXiv')
        doi = best.get('externalIds', {}).get('DOI')

        url = None
        if arxiv_id:
            url = f"https://arxiv.org/abs/{arxiv_id}"
        elif doi:
            url = f"https://doi.org/{doi}"

        return {
            'title': best.get('title'),
            'url': url,
            'authors': ', '.join(a.get('name', '') for a in best.get('authors', [])[:5]),
            'year': best.get('year'),
            'source': 'semantic_scholar',
            'similarity': title_similarity(codex_paper['title'], best.get('title', ''))
        }

    return None


def fix_paper_entry(codex_paper: dict, claude_papers: list[dict], verbose: bool = True, use_websearch: bool = True) -> dict:
    """Fix a single paper entry using available sources."""
    fixed = codex_paper.copy()
    issues = []
    recovery_source = None

    url_broken = not is_valid_paper_url(codex_paper.get('url', ''))
    title_suspicious = (
        len(codex_paper.get('title', '')) < 15 or
        'Unknown' in codex_paper.get('title', '') or
        codex_paper.get('title', '').lower() in ['mcp', 'neural information processing systems']
    )

    if url_broken or title_suspicious:
        # Try claude match first
        claude_match = find_claude_match(codex_paper, claude_papers)

        if claude_match:
            if url_broken and is_valid_paper_url(claude_match.get('url', '')):
                fixed['url'] = claude_match['url']
                issues.append('url_fixed_claude')
            if title_suspicious and claude_match.get('title'):
                fixed['title'] = claude_match['title']
                issues.append('title_fixed_claude')
            if not fixed.get('github_url') and claude_match.get('github_url'):
                fixed['github_url'] = claude_match['github_url']
                issues.append('github_added_claude')
            recovery_source = 'claude'
            if verbose:
                print(f"    [Claude] #{codex_paper['rank']}: {codex_paper['title'][:40]}...")
        elif use_websearch:
            # Try web search recovery
            recovered = recover_paper_info(codex_paper, verbose)
            if recovered and recovered.get('similarity', 0) > 0.4:
                if url_broken and recovered.get('url'):
                    fixed['url'] = recovered['url']
                    issues.append('url_fixed_websearch')
                if title_suspicious and recovered.get('title'):
                    fixed['title'] = recovered['title']
                    issues.append('title_fixed_websearch')
                recovery_source = 'semantic_scholar'
                if verbose:
                    print(f"    [WebSearch] #{codex_paper['rank']}: {recovered.get('title', '')[:40]}...")
            else:
                if url_broken:
                    issues.append('url_unrecoverable')
                if title_suspicious:
                    issues.append('title_unrecoverable')
                if verbose:
                    print(f"    [FAILED] #{codex_paper['rank']}: {codex_paper['title'][:40]}...")
        else:
            # Web search disabled
            if url_broken:
                issues.append('url_unrecoverable')
            if title_suspicious:
                issues.append('title_unrecoverable')
            if verbose:
                print(f"    [NO-MATCH] #{codex_paper['rank']}: {codex_paper['title'][:40]}...")

    fixed['_fix_issues'] = issues
    fixed['_recovery_source'] = recovery_source

    return fixed


def rewrite_report(report_path: Path, original_papers: list[dict], fixed_papers: list[dict]):
    """Rewrite the report with fixed paper information."""
    content = report_path.read_text()

    for orig, fixed in zip(original_papers, fixed_papers):
        if not fixed.get('_fix_issues'):
            continue

        # Replace URL if fixed
        if 'url_fixed_claude' in fixed['_fix_issues'] or 'url_fixed_websearch' in fixed['_fix_issues']:
            if orig.get('url') and fixed.get('url'):
                content = content.replace(orig['url'], fixed['url'])

        # Replace title if fixed
        if 'title_fixed_claude' in fixed['_fix_issues'] or 'title_fixed_websearch' in fixed['_fix_issues']:
            if orig.get('title') and fixed.get('title'):
                # Replace in header
                old_header = f"## {orig['rank']}. {orig['title']}"
                new_header = f"## {fixed['rank']}. {fixed['title']}"
                content = content.replace(old_header, new_header)

    report_path.write_text(content)


def process_topic(topic_num: str, dry_run: bool = False, verbose: bool = True, use_websearch: bool = True) -> dict:
    """Process a single topic and fix its codex report."""
    codex_name = TOPIC_NAMES.get(topic_num)
    claude_name = CLAUDE_TOPIC_NAMES.get(topic_num)

    if not codex_name:
        return {"error": f"Unknown topic: {topic_num}"}

    codex_path = CODEX_REPORTS / f"{codex_name}_influential_papers.md"
    claude_path = CLAUDE_REPORTS / f"{claude_name}_influential_papers.md"

    if not codex_path.exists():
        return {"error": f"Codex report not found: {codex_path.name}"}

    if verbose:
        print(f"\n{'='*60}")
        print(f"Topic {topic_num}: {codex_name}")
        print(f"{'='*60}")

    # Parse reports
    codex_papers = parse_codex_report(codex_path)
    claude_papers = parse_claude_report(claude_path) if claude_path.exists() else []

    if verbose:
        print(f"  Codex papers: {len(codex_papers)}")
        print(f"  Claude papers: {len(claude_papers)}")

    # Fix each paper
    fixed_papers = []
    stats = {
        "total": len(codex_papers),
        "url_broken": 0,
        "url_fixed_claude": 0,
        "url_fixed_websearch": 0,
        "url_unrecoverable": 0,
        "title_fixed_claude": 0,
        "title_fixed_websearch": 0,
        "github_added": 0,
    }

    for paper in codex_papers:
        if not is_valid_paper_url(paper.get('url', '')):
            stats["url_broken"] += 1

        fixed = fix_paper_entry(paper, claude_papers, verbose, use_websearch)
        fixed_papers.append(fixed)

        for issue in fixed.get('_fix_issues', []):
            if issue in stats:
                stats[issue] += 1

    # Rewrite report if not dry run
    if not dry_run:
        rewrite_report(codex_path, codex_papers, fixed_papers)
        if verbose:
            print(f"\n  Report updated: {codex_path.name}")

    if verbose:
        print(f"\n  Summary:")
        print(f"    URLs broken: {stats['url_broken']}")
        print(f"    Fixed via Claude: {stats['url_fixed_claude']}")
        print(f"    Fixed via WebSearch: {stats['url_fixed_websearch']}")
        print(f"    Unrecoverable: {stats['url_unrecoverable']}")

    return {
        "topic": topic_num,
        "topic_name": codex_name,
        "stats": stats,
        "papers": [
            {
                "rank": p["rank"],
                "title": p["title"],
                "issues": p.get("_fix_issues", []),
                "source": p.get("_recovery_source"),
            }
            for p in fixed_papers
            if p.get("_fix_issues")
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="Fix quality issues in codex reports")
    parser.add_argument("--topic", type=str, help="Topic number (01-20)")
    parser.add_argument("--all", action="store_true", help="Process all topics")
    parser.add_argument("--dry-run", action="store_true", help="Don't modify files")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    parser.add_argument("--no-websearch", action="store_true", help="Skip Semantic Scholar API (faster, Claude-only matching)")
    args = parser.parse_args()

    if args.all:
        print("Processing all topics...")
        if args.dry_run:
            print("(DRY RUN - no files will be modified)")

        all_results = []
        total_stats = {
            "total": 0,
            "url_broken": 0,
            "url_fixed_claude": 0,
            "url_fixed_websearch": 0,
            "url_unrecoverable": 0,
        }

        for topic_num in TOPIC_NAMES.keys():
            result = process_topic(topic_num, dry_run=args.dry_run, verbose=not args.quiet, use_websearch=not args.no_websearch)
            all_results.append(result)

            if "stats" in result:
                for key in total_stats:
                    total_stats[key] += result["stats"].get(key, 0)

        # Save quality report
        report = {
            "timestamp": datetime.now().isoformat(),
            "dry_run": args.dry_run,
            "summary": total_stats,
            "by_topic": {r["topic"]: r for r in all_results if "stats" in r},
            "manual_review_needed": [
                {"topic": r["topic"], "rank": p["rank"], "title": p["title"]}
                for r in all_results
                for p in r.get("papers", [])
                if "url_unrecoverable" in p.get("issues", [])
            ]
        }

        report_path = CODEX_REPORTS.parent / "quality_fix_report.json"
        report_path.write_text(json.dumps(report, indent=2))
        print(f"\n{'='*60}")
        print("OVERALL SUMMARY")
        print(f"{'='*60}")
        print(f"  Total papers: {total_stats['total']}")
        print(f"  URLs broken: {total_stats['url_broken']}")
        print(f"  Fixed via Claude: {total_stats['url_fixed_claude']}")
        print(f"  Fixed via WebSearch: {total_stats['url_fixed_websearch']}")
        print(f"  Unrecoverable: {total_stats['url_unrecoverable']}")
        print(f"\n  Quality report saved: {report_path.relative_to(PROJECT_ROOT)}")

    elif args.topic:
        topic_num = args.topic.zfill(2)
        process_topic(topic_num, dry_run=args.dry_run, verbose=not args.quiet, use_websearch=not args.no_websearch)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
