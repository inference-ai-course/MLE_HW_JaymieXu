import json
import time
from typing import Dict, List
from urllib.parse import urlencode

import settings as cfg

def normalize_title(s: str | None) -> str:
    if not s:
        return ""

    # Collapse all whitespace (including \r, \n, tabs) to single spaces.
    return " ".join(s.replace("\r", "\n").split())


def arxiv_search(query: str, total: int = 50, per_request: int = 50) -> List[Dict]:
    import feedparser
    out: List[Dict] = []
    start           = 0

    while len(out) < total:
        n = min(per_request, total - len(out))
        params = {
            "search_query": query,
            "start":        start,
            "max_results":  n,
            "sortBy":       "submittedDate",
            "sortOrder":    "descending",
        }

        qs = urlencode(params)
        feed = feedparser.parse(f"{cfg.ARXIV_API}?{qs}")

        entries = getattr(feed, "entries", []) or []
        got = len(entries)
        if got == 0:
            # nothing more to fetch
            break

        for entry in entries:
            # find a pdf link (prefer rel=related when type=application/pdf)
            pdf_url = None

            # PDF find
            for link in getattr(entry, "links", []):
                if getattr(link, "type", "") != "application/pdf":
                    continue

                # prefer the canonical one
                if getattr(link, "rel", "") == "related":
                    pdf_url = link.href
                    break

                # otherwise remember the FIRST pdf we saw (don’t overwrite later)
                if pdf_url is None:
                    pdf_url = link.href

            out.append({
                "id":               entry.id,
                "title":            normalize_title(entry.title),
                "authors":          [a.name for a in getattr(entry, "authors", [])],
                "published":        entry.published,
                "updated":          getattr(entry, "updated", entry.published),
                "summary":          entry.summary.strip(),
                "pdf_url":          pdf_url,
                "primary_category": entry.tags[0]["term"] if entry.tags else None,
            })

        # advance by what we actually got
        start += got

        # be polite to arXiv
        time.sleep(3)

    return out[:total]

def run_fetch_arxiv(total: int, per_request: int = 50, query: str = "cat:cs.CL"):
    results = arxiv_search(query, total=total, per_request=per_request)

    seen_ids = set()

    # Add existing pdfs if this is not first run
    if cfg.META_FILE.exists():
        for line in cfg.META_FILE.read_text(encoding="utf-8").splitlines():
            try:
                seen_ids.add(json.loads(line)["id"])
            except Exception:
                pass

    with cfg.META_FILE.open("a", encoding="utf-8") as mf:
        for rec in results:
            if rec["id"] in seen_ids:
                continue
            