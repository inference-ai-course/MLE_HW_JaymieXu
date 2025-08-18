from __future__ import annotations

from contextlib import asynccontextmanager
from typing import List, Optional, Any, Dict, cast

import json
import sqlite3
import faiss
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import settings as cfg

from sentence_transformers import SentenceTransformer
import torch
import numpy as np


# ---------------------------
# Pydantic models (I/O types)
# ---------------------------
class SearchRequest(BaseModel):
    query:     str = Field(..., description="Natural language search query")
    k:         int = Field(5, ge=1, le=50, description="Top-k results to return")
    min_score: float = Field(
        default=cfg.MIN_SCORE, ge=0, le=1,
        description="Cosine cutoff (server default)."
    )


class SearchHit(BaseModel):
    rank:     int
    score:    float
    chunk_id: str
    doc_id:   Optional[str] = None
    page:     Optional[int] = None
    title:    Optional[str] = None
    text:     Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    k:     int
    hits:  List[SearchHit]


# ------------------------------------------------
# Application lifespan: load once, reuse per call
# ------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    model_name = cfg.EMBED_MODEL
    metric = "cosine"

    # 1) load manifest to pick the same model the index used (fallback to cfg)
    try:
        mani       = json.loads(cfg.MANIFEST_PATH.read_text(encoding="utf-8"))
        model_name = mani.get("embed_model", model_name)
        metric     = mani.get("metric", metric)
    except Exception:
        # If manifest missing, we'll still proceed with cfg.EMBED_MODEL
        pass

    # 2) load embedder (prefer GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = SentenceTransformer(model_name, device=device)

    # 3) read FAISS index
    if not cfg.FAISS_INDEX_PATH.exists():
        raise RuntimeError(
            f"FAISS index not found at {cfg.FAISS_INDEX_PATH}. "
            f"Run your build step (run_faiss) first."
        )

    index = faiss.read_index(str(cfg.FAISS_INDEX_PATH))

    # 4) load sidecar (id-aligned with FAISS vectors)
    if not cfg.SIDE_CAR_PATH.exists():
        raise RuntimeError(
            f"Sidecar not found at {cfg.SIDE_CAR_PATH}. "
            f"Run your build step (run_faiss) to generate it."
        )

    side: List[Dict[str, Any]] = []
    conn   = sqlite3.connect(cfg.SIDE_CAR_PATH)
    cursor = conn.execute("SELECT chunk_id, doc_id, page, title FROM chunk_meta ORDER BY rowid")
    
    for row in cursor.fetchall():
        side.append({
            "chunk_id": row[0],
            "doc_id":   row[1],
            "page":     row[2],
            "title":    row[3]
        })
        
    conn.close()
    
    # CHANGE: Build chunk_text map from SQLite instead of JSONL
    chunk_text: Dict[str, str] = {}
    chunks_conn   = sqlite3.connect(cfg.CHUNKS_OUT)
    chunks_cursor = chunks_conn.execute("SELECT chunk_id, text FROM chunks WHERE text IS NOT NULL AND text != ''")
    
    for row in chunks_cursor.fetchall():
        cid = row[0]
        txt = row[1]
        if cid and txt:
            chunk_text[cid] = txt
            
    chunks_conn.close()

    # quick consistency check (warn only)
    if index.ntotal != len(side):
        print(f"[warn] FAISS vectors = {index.ntotal}, sidecar lines = {len(side)}. "
              f"Search will still run, but audit your build step.")

    # stash in app.state for fast access
    state = cast(Any, getattr(app, "state"))
    state.model      = model
    state.index      = index
    state.side       = side
    state.chunk_text = chunk_text
    state.metric     = metric
    state.device     = device

    print(f"[startup] model={model_name} device={device} vectors={index.ntotal} metric={metric}")
    yield


# ------------------
# FastAPI app object
# ------------------
app = FastAPI(title="RAG Search API", version="0.1.0", lifespan=lifespan)


# ----------------
# Helper functions
# ----------------
def _embed_query(model: SentenceTransformer, text: str) -> np.ndarray:
    vec = model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine via IP
        show_progress_bar=False,
        batch_size=1,
    ).astype("float32")

    return vec


def _format_hits(
        ids: np.ndarray,
        scores: np.ndarray,
        side: List[dict],
        k: int,
        chunk_text: Dict[str, str]) -> List[SearchHit]:
    out: List[SearchHit] = []
    id_row               = ids[0].tolist()
    sc_row               = scores[0].tolist()

    for rank, (i, s) in enumerate(zip(id_row, sc_row), start=1):
        if i < 0 or i >= len(side):
            # FAISS may return -1 when not enough neighbors exist
            continue

        meta = side[i]
        out.append(
            SearchHit(
                rank=rank,
                score=float(s),
                chunk_id=meta.get("chunk_id", ""),
                doc_id=meta.get("doc_id"),
                page=meta.get("page"),
                title=meta.get("title"),
                text=chunk_text.get(meta.get("chunk_id", ""), None),
            )
        )

        if len(out) >= k:
            break

    return out


# -----------
# Endpoints
# -----------
@app.get("/healthz")
def healthz() -> dict:
    """Simple health check endpoint."""
    state = cast(Any, getattr(app, "state"))
    return {"ok": True, "vectors": int(state.index.ntotal), "device": state.device}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    """
    Semantic search.
    - Embed the query with the same model used to build the index
    - Search FAISS (cosine via inner product)
    - Map ids to metadata from sidecar and return top-k hits
    """
    state = cast(Any, getattr(app, "state"))

    if state.index is None or state.model is None:
        raise HTTPException(status_code=503, detail="Index not loaded. Try again shortly.")

    qv          = _embed_query(state.model, req.query)
    scores, ids = state.index.search(qv, req.k)
    hits        = _format_hits(ids, scores, state.side, req.k, state.chunk_text)

    # Perform cut off
    hits = [h for h in hits if h.score >= req.min_score][:req.k]

    return SearchResponse(query=req.query, k=req.k, hits=hits)


# -------------------------
# Local dev entry point
# -------------------------
if __name__ == "__main__":
    # Note: 'reload=True' is handy during development.
    uvicorn.run("search_api:app", host="127.0.0.1", port=8000, reload=True)
