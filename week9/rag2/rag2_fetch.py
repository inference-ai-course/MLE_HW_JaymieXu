#!/usr/bin/env python3
"""
Enhanced ArXiv fetcher that searches by keywords instead of categories.
Provides better control over data quality by searching specific topics.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Set
from urllib.parse import urlencode

import feedparser
import requests

from rag2_config import cfg, get_rag2_config_path


# ArXiv API endpoint
ARXIV_API = "http://export.arxiv.org/api/query"

# Default keywords for high-quality research papers
DEFAULT_KEYWORDS = [
    "cyber security"
]


def normalize_title(s: str | None) -> str:
    """Normalize paper title by collapsing whitespace"""
    if not s:
        return ""
    return " ".join(s.replace("\r", "\n").split())


def extract_arxiv_id(id_field: str) -> str:
    """Extract ArXiv ID from full URL"""
    return id_field.strip().split("/")[-1]


def arxiv_search_keyword(keyword: str, max_results: int = 10, sort_by: str = "relevance", min_year: int = 2020) -> List[Dict]:
    """
    Search ArXiv for papers matching a specific keyword.

    Args:
        keyword: Search term (e.g., "machine learning")
        max_results: Maximum papers to retrieve for this keyword
        sort_by: Sort order - "relevance", "lastUpdatedDate", or "submittedDate"
        min_year: Minimum publication year (e.g., 2020)

    Returns:
        List of paper metadata dictionaries
    """
    print(f"[INFO] Searching for '{keyword}' (max {max_results} papers)...")

    results = []
    start = 0
    per_request = min(50, max_results)  # ArXiv API limit

    while len(results) < max_results:
        batch_size = min(per_request, max_results - len(results))

        # Build search query - search in title and abstract
        # Handle hyphenated terms and improve search flexibility
        if "-" in keyword:
            # For hyphenated terms, try multiple variations
            no_hyphen = keyword.replace("-", " ")  # "zero-day" -> "zero day"
            underscore = keyword.replace("-", "_")  # "zero-day" -> "zero_day"
            search_query = f'ti:"{keyword}" OR abs:"{keyword}" OR ti:"{no_hyphen}" OR abs:"{no_hyphen}" OR ti:{keyword} OR abs:{keyword}'
        else:
            search_query = f'ti:"{keyword}" OR abs:"{keyword}"'

        params = {
            "search_query": search_query,
            "start": start,
            "max_results": batch_size,
            "sortBy": sort_by,
            "sortOrder": "descending",
        }

        url = f"{ARXIV_API}?{urlencode(params)}"

        try:
            feed = feedparser.parse(url)
            entries = getattr(feed, "entries", [])

            if not entries:
                print(f"[INFO] No more results for '{keyword}'")
                break

            for entry in entries:
                # Check publication year first to avoid processing old papers
                try:
                    # ArXiv published date format: "2023-12-15T18:30:02Z"
                    pub_year = int(entry.published.split('-')[0])
                    if pub_year < min_year:
                        continue  # Skip papers older than min_year
                except (ValueError, AttributeError):
                    # If we can't parse the year, skip this paper
                    continue

                # Extract PDF URL
                pdf_url = None
                for link in getattr(entry, "links", []):
                    if getattr(link, "type", "") == "application/pdf":
                        if getattr(link, "rel", "") == "related":
                            pdf_url = link.href
                            break
                        elif pdf_url is None:
                            pdf_url = link.href

                # Extract paper metadata
                paper = {
                    "id": entry.id,
                    "arxiv_id": extract_arxiv_id(entry.id),
                    "title": normalize_title(entry.title),
                    "authors": [author.name for author in getattr(entry, "authors", [])],
                    "published": entry.published,
                    "updated": getattr(entry, "updated", entry.published),
                    "summary": entry.summary.strip(),
                    "pdf_url": pdf_url,
                    "primary_category": entry.tags[0]["term"] if entry.tags else None,
                    "search_keyword": keyword,
                    "year": pub_year,
                }

                results.append(paper)

            start += len(entries)

            # Be polite to ArXiv API
            time.sleep(2)

        except Exception as e:
            print(f"[ERROR] Failed to search '{keyword}': {e}")
            break

    print(f"[OK] Found {len(results)} papers for '{keyword}'")
    return results


def download_pdf(url: str, dest_path: Path, retries: int = 3) -> bool:
    """
    Download PDF from URL to destination path.

    Args:
        url: PDF download URL
        dest_path: Local file path to save PDF
        retries: Number of retry attempts

    Returns:
        True if download successful, False otherwise
    """
    # Skip if file already exists and is not empty
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return True

    for attempt in range(1, retries + 1):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; Academic-Research-Bot/1.0)'
            }

            with requests.get(url, headers=headers, stream=True, timeout=30) as response:
                response.raise_for_status()

                # Create parent directory if needed
                dest_path.parent.mkdir(parents=True, exist_ok=True)

                with open(dest_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

            # Verify download
            if dest_path.exists() and dest_path.stat().st_size > 0:
                return True

        except Exception as e:
            print(f"[WARN] Download attempt {attempt} failed for {url}: {e}")
            if dest_path.exists():
                dest_path.unlink()  # Remove corrupted file

            if attempt < retries:
                time.sleep(2 * attempt)  # Exponential backoff

    return False


def fetch_papers_by_keywords(keywords: List[str] = None,
                            papers_per_keyword: int = 10,
                            sort_by: str = "relevance",
                            min_year: int = 2020) -> Dict[str, List[Dict]]:
    """
    Fetch papers from ArXiv for multiple keywords.

    Args:
        keywords: List of search keywords
        papers_per_keyword: Maximum papers to fetch per keyword
        sort_by: Sort order for results
        min_year: Minimum publication year (e.g., 2020)

    Returns:
        Dictionary mapping keywords to lists of paper metadata
    """
    if keywords is None:
        keywords = DEFAULT_KEYWORDS

    print(f"[INFO] Fetching papers for {len(keywords)} keywords...")
    print(f"[INFO] Max {papers_per_keyword} papers per keyword")

    all_results = {}
    seen_arxiv_ids: Set[str] = set()

    for i, keyword in enumerate(keywords, 1):
        print(f"\n[{i}/{len(keywords)}] Processing: {keyword}")

        papers = arxiv_search_keyword(keyword, papers_per_keyword, sort_by, min_year)

        # Remove duplicates across keywords
        unique_papers = []
        for paper in papers:
            arxiv_id = paper["arxiv_id"]
            if arxiv_id not in seen_arxiv_ids:
                seen_arxiv_ids.add(arxiv_id)
                unique_papers.append(paper)

        all_results[keyword] = unique_papers

        if unique_papers:
            print(f"[OK] Added {len(unique_papers)} unique papers for '{keyword}'")
        else:
            print(f"[SKIP] No unique papers for '{keyword}'")

    total_papers = sum(len(papers) for papers in all_results.values())
    print(f"\n[SUMMARY] Collected {total_papers} unique papers across {len(keywords)} keywords")

    return all_results


def download_papers(results: Dict[str, List[Dict]], max_downloads: int = None) -> None:
    """
    Download PDFs for fetched papers.

    Args:
        results: Results from fetch_papers_by_keywords()
        max_downloads: Maximum total downloads (None for no limit)
    """
    raw_dir = get_rag2_config_path("raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Flatten all papers and sort by relevance/date
    all_papers = []
    for keyword, papers in results.items():
        all_papers.extend(papers)

    if max_downloads:
        all_papers = all_papers[:max_downloads]

    print(f"\n[INFO] Downloading {len(all_papers)} PDFs to {raw_dir}")

    successful = 0
    failed = 0

    for i, paper in enumerate(all_papers, 1):
        arxiv_id = paper["arxiv_id"]
        title = paper["title"][:50] + "..." if len(paper["title"]) > 50 else paper["title"]

        print(f"[{i}/{len(all_papers)}] {arxiv_id}: {title}")

        if not paper["pdf_url"]:
            print(f"[SKIP] No PDF URL for {arxiv_id}")
            failed += 1
            continue

        # Generate safe filename
        safe_filename = f"{arxiv_id.replace('/', '_')}.pdf"
        pdf_path = raw_dir / safe_filename


        if download_pdf(paper["pdf_url"], pdf_path):
            print(f"[OK] Downloaded {arxiv_id}")
            successful += 1
        else:
            print(f"[FAIL] Could not download {arxiv_id}")
            failed += 1

        # Rate limiting
        time.sleep(1)

    print(f"\n[SUMMARY] Downloads: {successful} successful, {failed} failed")




def main():
    """Main execution function"""
    print("=== RAG2 Enhanced ArXiv Fetcher ===")

    # Custom keywords for your research domain
    keywords = [
        "cyber security",
        "network security",
        "cryptography",
        "data privacy",
        "authentication",
        "phishing",
        "ransomware",
        "malware",
        "botnet",
        "denial of service",
        "sql injection",
        "cross-site scripting",
        "zero-day",
        "insider threat",
        "intrusion detection",
        "threat intelligence",
        "vulnerability assessment",
        "penetration testing",
        "incident response",
        "cloud security",
        "iot security",
        "blockchain security",
        "adversarial machine learning",
        "privacy preserving machine learning",
        "digital forensics",
    ]

    # Fetch papers (only from 2020 onwards)
    results = fetch_papers_by_keywords(
        keywords=keywords,
        papers_per_keyword=3,  # Fewer but higher quality
        sort_by="relevance",
        min_year=2020  # Only papers from 2020 onwards
    )

    # Download PDFs (no limit)
    download_papers(results, max_downloads=None)

    print("\n=== Fetch Complete ===")


if __name__ == "__main__":
    main()