import json
import time
from urllib.parse import urlencode
import feedparser

import settings as cfg

def normalize_title(s: str | None) -> str:
    if not s:
        return ""
    return " ".join(s.replace("\r", "\n").split())

def fetch_arxiv_abstracts(query: str = "cat:cs.CL", total: int = 300, per_request: int = 50):
    """
    Fetch title and abstract from arXiv and write to JSONL file
    """
    results = []
    start = 0
    
    # Load existing IDs to avoid duplicates
    seen_ids = set()
    if cfg.META_FILE.exists():
        for line in cfg.META_FILE.read_text(encoding="utf-8").splitlines():
            try:
                seen_ids.add(json.loads(line)["id"])
            except Exception:
                pass
    
    while len(results) < total:
        n = min(per_request, total - len(results))
        params = {
            "search_query": query,
            "start": start,
            "max_results": n,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        
        qs = urlencode(params)
        feed = feedparser.parse(f"{cfg.ARXIV_API}?{qs}")
        
        entries = getattr(feed, "entries", []) or []
        if not entries:
            break
            
        for entry in entries:
            paper_id = entry.id
            if paper_id in seen_ids:
                continue
                
            paper_data = {
                "id": paper_id,
                "title": normalize_title(entry.title),
                "abstract": entry.summary.strip()
            }
            
            results.append(paper_data)
            seen_ids.add(paper_id)
        
        start += len(entries)
        time.sleep(3)  # Be polite to arXiv
    
    # Write to JSONL file
    with cfg.META_FILE.open("a", encoding="utf-8") as f:
        for paper in results:
            f.write(json.dumps(paper) + "\n")
    
    print(f"Fetched {len(results)} papers and saved to {cfg.META_FILE}")
    return results

if __name__ == "__main__":
    fetch_arxiv_abstracts()
            