#!/usr/bin/env python3
"""Find correct paper sources from arXiv and OpenReview for papers needing re-download."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from xml.etree import ElementTree

from dotenv import load_dotenv

from conference_pipeline.model_governance import default_model
from conference_pipeline.provider_receipts import (
    gemini_generate_content,
    infer_conference_workspace,
    openai_chat_completions_create,
)

# source_discovery.paper_source_ranking is the registry-governed task for this
# script's LLM-ranked paper-title/source matching.
DEFAULT_PAPER_SOURCE_RANKING_MODEL = default_model("source_discovery.paper_source_ranking")

load_dotenv()

# Try to import OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Try to import Gemini
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

if not OPENAI_AVAILABLE and not GEMINI_AVAILABLE:
    raise SystemExit("Missing: pip install openai or pip install google-genai")


def search_arxiv(query: str, max_results: int = 5) -> list[dict]:
    """Search arXiv for papers matching the query."""
    base_url = "http://export.arxiv.org/api/query"

    # Clean and encode query
    clean_query = re.sub(r'[^\w\s\-]', '', query)
    params = urllib.parse.urlencode({
        'search_query': f'all:{clean_query}',
        'start': 0,
        'max_results': max_results,
        'sortBy': 'relevance',
        'sortOrder': 'descending'
    })

    url = f"{base_url}?{params}"

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read().decode('utf-8')

        # Parse XML response
        root = ElementTree.fromstring(data)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}

        results = []
        for entry in root.findall('atom:entry', ns):
            title = entry.find('atom:title', ns)
            summary = entry.find('atom:summary', ns)

            # Get PDF link
            pdf_link = None
            for link in entry.findall('atom:link', ns):
                if link.get('title') == 'pdf':
                    pdf_link = link.get('href')
                    break

            # Get arXiv ID
            arxiv_id = None
            id_elem = entry.find('atom:id', ns)
            if id_elem is not None:
                # Extract ID from URL like http://arxiv.org/abs/2301.12345v1
                match = re.search(r'arxiv\.org/abs/(.+)', id_elem.text)
                if match:
                    arxiv_id = match.group(1)

            if title is not None:
                results.append({
                    'source': 'arxiv',
                    'title': ' '.join(title.text.split()),  # Clean whitespace
                    'abstract': summary.text[:500] if summary is not None else '',
                    'arxiv_id': arxiv_id,
                    'pdf_url': pdf_link,
                    'page_url': f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None
                })

        return results
    except Exception as e:
        print(f"  arXiv search error: {e}")
        return []


def search_openreview(query: str, max_results: int = 5) -> list[dict]:
    """Search OpenReview for papers matching the query."""
    # OpenReview API endpoint
    base_url = "https://api2.openreview.net/notes/search"

    # Clean query
    clean_query = re.sub(r'[^\w\s\-]', ' ', query)

    params = urllib.parse.urlencode({
        'query': clean_query,
        'limit': max_results,
        'invitation': 'NeurIPS.cc/2024/Conference/-/Submission'  # Try NeurIPS 2024
    })

    url = f"{base_url}?{params}"

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode('utf-8'))

        results = []
        for note in data.get('notes', [])[:max_results]:
            content = note.get('content', {})
            title = content.get('title', {})
            if isinstance(title, dict):
                title = title.get('value', '')

            abstract = content.get('abstract', {})
            if isinstance(abstract, dict):
                abstract = abstract.get('value', '')

            forum_id = note.get('forum', note.get('id', ''))

            # Try to get PDF
            pdf_url = None
            if 'pdf' in content:
                pdf_ref = content['pdf']
                if isinstance(pdf_ref, dict):
                    pdf_ref = pdf_ref.get('value', '')
                if pdf_ref:
                    pdf_url = f"https://openreview.net{pdf_ref}" if pdf_ref.startswith('/') else pdf_ref

            if title:
                results.append({
                    'source': 'openreview',
                    'title': title,
                    'abstract': abstract[:500] if abstract else '',
                    'forum_id': forum_id,
                    'pdf_url': pdf_url,
                    'page_url': f"https://openreview.net/forum?id={forum_id}" if forum_id else None
                })

        return results
    except Exception as e:
        print(f"  OpenReview search error: {e}")
        return []


def simple_title_match(expected_title: str, search_results: list[dict]) -> dict | None:
    """Simple heuristic matching based on title keywords."""
    if not search_results:
        return None

    expected_lower = expected_title.lower()

    # Extract key terms from expected title
    # Find acronyms (2-8 uppercase letters/numbers, allowing hyphenated versions like DPLM-2)
    acronyms = re.findall(r'\b[A-Z][A-Z0-9\-]{1,7}\b', expected_title)

    # Also extract the first word if it looks like a paper name (capitalized)
    first_words = expected_title.split(':')[0].strip().split()
    if first_words and first_words[0][0].isupper():
        paper_name = first_words[0]
        if paper_name not in acronyms:
            acronyms.append(paper_name)

    # Find important words (longer than 4 chars)
    words = [w.lower() for w in re.findall(r'\b\w{5,}\b', expected_title)]

    best_score = 0
    best_result = None

    for result in search_results:
        result_title = result['title']
        result_lower = result_title.lower()
        score = 0

        # Check for exact title prefix match (highest priority)
        expected_prefix = expected_lower.split(':')[0].strip()
        result_prefix = result_lower.split(':')[0].strip()
        if expected_prefix == result_prefix:
            score += 100  # Near-exact match

        # Check for acronym matches (high weight)
        for acronym in acronyms:
            # Exact match in title
            if acronym in result_title:
                score += 20
            # Case-insensitive match
            elif acronym.lower() in result_lower:
                score += 15

        # Check for word matches
        for word in words:
            if word in result_lower:
                score += 2

        # Bonus for source preference (arXiv slightly preferred)
        if result['source'] == 'arxiv':
            score += 0.5

        if score > best_score:
            best_score = score
            best_result = result

    # Require minimum score (at least one strong match)
    if best_score >= 10 and best_result:
        confidence = "high" if best_score >= 50 else "medium" if best_score >= 20 else "low"
        return {
            **best_result,
            'match_confidence': confidence,
            'match_reason': f"Title keyword match (score: {best_score})"
        }

    return None


def rank_results_with_llm(
    expected_title: str,
    search_results: list[dict],
    model_name: str = DEFAULT_PAPER_SOURCE_RANKING_MODEL
) -> dict | None:
    """Use LLM (OpenAI or Gemini) to rank search results and find best match."""

    if not search_results:
        return None

    # First try simple heuristic matching
    simple_match = simple_title_match(expected_title, search_results)
    if simple_match and simple_match['match_confidence'] == 'high':
        return simple_match

    # Build results summary
    results_text = []
    for i, r in enumerate(search_results):
        results_text.append(f"""
Result {i+1} [{r['source']}]:
  Title: {r['title']}
  Abstract: {r['abstract'][:300]}...
  URL: {r.get('page_url', 'N/A')}""")

    prompt = f"""Match this paper title to search results.

Expected: {expected_title}

Results:
{"".join(results_text[:5])}

Which result (1-{min(len(search_results), 5)}) best matches? Return JSON:
{{"best_match_index": NUMBER_OR_NULL, "confidence": "high/medium/low", "reason": "brief"}}

JSON only:"""

    try:
        # Use OpenAI if model starts with gpt, otherwise use Gemini
        if model_name.startswith('gpt') or model_name.startswith('o1') or model_name.startswith('o3'):
            if not OPENAI_AVAILABLE:
                print(f"  OpenAI not available, using heuristic match")
                return simple_match

            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            response = openai_chat_completions_create(client, caller=__file__,
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_completion_tokens=500
            )
            text = response.choices[0].message.content.strip()
        else:
            if not GEMINI_AVAILABLE:
                print(f"  Gemini not available, using heuristic match")
                return simple_match

            client = genai.Client(
                api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            )
            response = gemini_generate_content(client, caller=__file__,
                model=model_name,
                contents=prompt,
                config={"temperature": 0.1, "max_output_tokens": 500}
            )

            # Check if response has text
            if not hasattr(response, 'text') or not response.text:
                return simple_match
            text = response.text.strip()

        # Clean markdown
        if "```" in text:
            match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
            if match:
                text = match.group(1).strip()

        result = json.loads(text)

        if result.get('best_match_index'):
            idx = result['best_match_index'] - 1  # Convert to 0-indexed
            if 0 <= idx < len(search_results):
                return {
                    **search_results[idx],
                    'match_confidence': result.get('confidence', 'unknown'),
                    'match_reason': result.get('reason', '')
                }

        return simple_match  # Fall back to simple match
    except Exception as e:
        print(f"  LLM ranking error: {e}")
        return simple_match  # Fall back to simple match


def find_paper_sources(
    verification_path: Path,
    output_path: Path,
    limit: int = None,
    skip_existing: bool = True,
    model_name: str = DEFAULT_PAPER_SOURCE_RANKING_MODEL
) -> dict:
    """Find sources for papers that need re-downloading."""

    with open(verification_path) as f:
        data = json.load(f)

    papers = data.get('needs_redownload', [])
    if limit:
        papers = papers[:limit]

    print(f"Finding sources for {len(papers)} papers...", flush=True)

    # Load existing results if continuing
    results = {
        'found': [],
        'not_found': [],
        'errors': []
    }

    existing_titles = set()
    if skip_existing and output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
            results = existing
            existing_titles = {p['expected_title'] for p in existing.get('found', [])}
            existing_titles.update(p['expected_title'] for p in existing.get('not_found', []))
            print(f"Loaded {len(existing_titles)} existing results", flush=True)

    for i, paper in enumerate(papers):
        expected_title = paper['expected_title']

        if expected_title in existing_titles:
            print(f"[{i+1}/{len(papers)}] Skipping (exists): {expected_title[:50]}...", flush=True)
            continue

        print(f"[{i+1}/{len(papers)}] Searching: {expected_title[:50]}...", flush=True)

        try:
            # Search both sources
            arxiv_results = search_arxiv(expected_title)
            time.sleep(0.5)  # Rate limit

            openreview_results = search_openreview(expected_title)
            time.sleep(0.5)

            all_results = arxiv_results + openreview_results

            if not all_results:
                print(f"  -> No results found", flush=True)
                results['not_found'].append({
                    **paper,
                    'search_attempted': True,
                    'arxiv_results': 0,
                    'openreview_results': 0
                })
                continue

            print(f"  -> Found {len(arxiv_results)} arXiv, {len(openreview_results)} OpenReview results", flush=True)

            # Show top candidates
            for j, r in enumerate(all_results[:3]):
                print(f"     Candidate {j+1}: [{r['source']}] {r['title'][:60]}...", flush=True)

            # Rank with LLM
            best_match = rank_results_with_llm(expected_title, all_results, model_name)

            if best_match:
                print(f"  -> MATCH [{best_match['source']}]: {best_match['title'][:50]}... ({best_match['match_confidence']})", flush=True)
                results['found'].append({
                    **paper,
                    'new_source': best_match['source'],
                    'new_title': best_match['title'],
                    'new_pdf_url': best_match.get('pdf_url'),
                    'new_page_url': best_match.get('page_url'),
                    'match_confidence': best_match['match_confidence'],
                    'match_reason': best_match['match_reason']
                })
            else:
                print(f"  -> No good match found", flush=True)
                results['not_found'].append({
                    **paper,
                    'search_attempted': True,
                    'arxiv_results': len(arxiv_results),
                    'openreview_results': len(openreview_results),
                    'candidates': [r['title'] for r in all_results[:3]]
                })

        except Exception as e:
            print(f"  -> ERROR: {e}", flush=True)
            results['errors'].append({
                **paper,
                'error': str(e)
            })

        # Save progress periodically
        if (i + 1) % 10 == 0:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"  [Progress saved: {len(results['found'])} found, {len(results['not_found'])} not found]", flush=True)

        time.sleep(1)  # Rate limiting

    # Final summary
    print(f"\n=== SUMMARY ===", flush=True)
    print(f"Found: {len(results['found'])}", flush=True)
    print(f"Not found: {len(results['not_found'])}", flush=True)
    print(f"Errors: {len(results['errors'])}", flush=True)

    # Save final results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="Find paper sources from arXiv/OpenReview")
    parser.add_argument(
        "verification_path",
        type=Path,
        help="Path to verification_full.json"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("paper_sources.json"),
        help="Output path for found sources"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        help="Limit number of papers to search"
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Don't skip papers already in output file"
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_PAPER_SOURCE_RANKING_MODEL,
        help='LLM model to use for ranking (registry default: source_discovery.paper_source_ranking)'
    )
    args = parser.parse_args()

    # D-QUICK-260715-rz1: this script has no --workspace flag, so derive one
    # from the verification_path positional arg to avoid falling back to
    # Path.cwd() (repo root).
    inferred_workspace = infer_conference_workspace(args.verification_path)
    if inferred_workspace is not None:
        os.environ.setdefault("PROVIDER_RECEIPT_WORKSPACE", str(inferred_workspace))
    os.environ.setdefault("PROVIDER_RECEIPT_STAGE_ID", "find_paper_sources")
    os.environ.setdefault("PROVIDER_RECEIPT_CALLER", __file__)

    find_paper_sources(
        args.verification_path,
        args.output,
        args.limit,
        skip_existing=not args.no_skip,
        model_name=args.model
    )


if __name__ == "__main__":
    main()
