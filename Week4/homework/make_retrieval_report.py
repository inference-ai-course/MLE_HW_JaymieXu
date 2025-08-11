from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Iterable, List, Dict, Any

import requests
import textwrap
import settings as cfg


DEF_QUERIES: List[str] = [
    "transformer attention mechanism",
    "beam search decoding",
    "contrastive learning for embeddings",
    "reinforcement learning for language models",
    "CTC loss",
]

API_BASE = os.getenv("SEARCH_API", "http://127.0.0.1:8000")


def _snippet(text: str, max_chars: int = 600) -> str:
    """Collapse whitespace and truncate for pretty Markdown."""
    s = " ".join((text or "").split())
    return s[:max_chars] + ("…" if len(s) > max_chars else "")


def _load_manifest() -> Dict[str, Any]:
    """Best-effort read of manifest for metadata (model name, dim, etc.)."""
    try:
        return json.loads(cfg.MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _health() -> Dict[str, Any]:
    try:
        r = requests.get(f"{API_BASE}/healthz", timeout=10)
        return r.json()
    except Exception:
        return {}


def _search(query: str, k: int, min_score: float) -> Dict[str, Any]:
    payload = {"query": query, "k": k, "min_score": min_score}

    r = requests.post(f"{API_BASE}/search", json=payload, timeout=60)
    r.raise_for_status()

    return r.json()


def _wrap(text: str, width: int = 110) -> str:
    # normalize spaces, then hard-wrap; also break very long tokens (CJK/equations)
    cleaned = " ".join((text or "").split())
    return textwrap.fill(
        cleaned,
        width=width,
        break_long_words=True,
        break_on_hyphens=False,
    )


def generate_report(
    queries:   Iterable[str],
    out_path:  str | Path = "retrieval_report.md",
    k:         int = 3,
    min_score: float = cfg.MIN_SCORE,
) -> Path:
    """
    Run queries against /search and write a Markdown report.
    Returns the output Path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mani   = _load_manifest()
    health = _health()

    lines: List[str] = ["# Retrieval Report\n", f"- generated: {dt.datetime.now().isoformat()}"]
    model_name       = mani.get("embed_model") or cfg.EMBED_MODEL
    lines.append(f"- model: `{model_name}`")

    if "vectors" in health:
        lines.append(f"- index size: {health['vectors']} vectors")
    lines.append(f"- parameters: k={k}, min_score={min_score:.2f}\n")

    for q in queries:
        lines.append(f"\n## Query: `{q}`\n")
        try:
            resp = _search(q, k=k, min_score=min_score)
        except Exception as e:
            lines.append(f"- ERROR calling API: `{e}`\n")
            continue

        hits = resp.get("hits", [])
        if not hits:
            lines.append("- No passages passed the score threshold.\n")
            continue

        for h in hits:
            title  = h.get("title") or "(untitled)"
            page   = h.get("page")
            score  = h.get("score", 0.0)
            text   = h.get("text") or ""
            lines.append(f"- **score={score:.3f} · p{page} · {title}**")
            lines.append("")
            wrapped = _wrap(text, width=110).splitlines()
            lines.extend([f"  > {ln}  " for ln in wrapped])

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote {out_path})")

    return out_path


def display_report(path: str | Path) -> None:
    """
    Render a Markdown report inside a Jupyter notebook cell.
    If not in a notebook, just prints the file path.
    """
    p = Path(path)

    try:
        from IPython.display import Markdown, display  # only available in notebooks
        display(Markdown(p.read_text(encoding="utf-8")))
    except Exception:
        print(f"(Open in editor) {p.resolve()}")

