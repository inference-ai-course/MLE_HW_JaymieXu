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
from rank_bm25 import BM25Okapi


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


class KeywordSearchRequest(BaseModel):
    query:     str = Field(..., description="Keyword search query")
    k:         int = Field(5, ge=1, le=50, description="Top-k results to return")
    min_score: float = Field(
        default=cfg.MIN_FTS_SCORE, ge=0, le=1,
        description="Minimum FTS5 relevance score"
    )


class KeywordSearchResponse(BaseModel):
    query: str
    k:     int
    hits:  List[SearchHit]


class HybridSearchRequest(BaseModel):
    query:        str   = Field(..., description="Search query for both semantic and keyword search")
    k:            int   = Field(5, ge=1, le=50, description="Top-k results to return")
    min_score:    float = Field(default=cfg.MIN_SCORE, ge=0, le=1, description="Minimum semantic score")
    semantic_k:   int   = Field(10, ge=1, le=100, description="Top-k results from semantic search for fusion")
    keyword_k:    int   = Field(10, ge=1, le=100, description="Top-k results from keyword search for fusion")
    rrf_constant: int   = Field(60, ge=1, le=100, description="RRF constant (typically 60)")


class HybridSearchResponse(BaseModel):
    query:  str
    k:      int
    hits:   List[SearchHit]  # Reuse existing SearchHit model
    method: str = "hybrid_rrf"  # ADD: Indicate fusion method used


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
    
    # Build chunk_text map from SQLite
    chunk_text: Dict[str, str] = {}
    chunks_conn   = sqlite3.connect(cfg.CHUNKS_OUT)
    chunks_cursor = chunks_conn.execute("SELECT chunk_id, text FROM chunks WHERE text IS NOT NULL AND text != ''")
    
    for row in chunks_cursor.fetchall():
        cid = row[0]
        txt = row[1]
        if cid and txt:
            chunk_text[cid] = txt
            
    chunks_conn.close()
    
    # Test FTS5 availability and cache connection info
    fts5_available = False
    try:
        test_conn = sqlite3.connect(cfg.CHUNKS_OUT)
        test_conn.execute("SELECT COUNT(*) FROM chunks_fts LIMIT 1").fetchone()
        fts5_available = True
        test_conn.close()
        print(f"[startup] FTS5 search available")
        
    except Exception as e:
        print(f"[warn] FTS5 not available: {e}")

    # Create BM25 index for re-ranking
    bm25_index = None
    chunk_id_to_text = {}
    
    try:
        bm25_conn = sqlite3.connect(cfg.CHUNKS_OUT)
        cursor = bm25_conn.execute("SELECT chunk_id, text FROM chunks WHERE text IS NOT NULL AND text != ''")
        
        docs = []
        chunk_ids = []
        for row in cursor.fetchall():
            chunk_id, text = row
            if text.strip():
                # Simple tokenization for BM25
                tokenized_text = text.lower().split()
                docs.append(tokenized_text)
                chunk_ids.append(chunk_id)
                chunk_id_to_text[chunk_id] = text
        
        if docs:
            bm25_index = BM25Okapi(docs)
            print(f"[startup] BM25 index created with {len(docs)} chunks for re-ranking")
        
        bm25_conn.close()
        
    except Exception as e:
        print(f"[warn] BM25 not available: {e}")

    # quick consistency check (warn only)
    if index.ntotal != len(side):
        print(f"[warn] FAISS vectors = {index.ntotal}, sidecar lines = {len(side)}. "
              f"Search will still run, but audit your build step.")

    # stash in app.state for fast access
    state = cast(Any, getattr(app, "state"))
    state.model          = model
    state.index          = index
    state.side           = side
    state.chunk_text     = chunk_text
    state.metric         = metric
    state.device         = device
    state.fts5_available = fts5_available
    state.bm25_index     = bm25_index
    state.bm25_chunk_ids = chunk_ids if bm25_index else []
    state.chunk_id_to_text = chunk_id_to_text

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


def _keyword_search(query: str, k: int) -> List[Dict[str, Any]]:
    """Perform FTS5 keyword search on chunks"""
    conn = sqlite3.connect(cfg.CHUNKS_OUT)

    try:
        # Use FTS5 MATCH query with rank-based scoring
        cursor = conn.execute('''
            SELECT chunk_id, doc_id, text, rank AS fts_score
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        ''', (query, k))

        results = []
        for row in cursor.fetchall():
            results.append({
                'chunk_id':  row[0],
                'doc_id':    row[1],
                'text':      row[2],
                'fts_score': abs(row[3]) if row[3] else 0.0  # Convert negative rank to positive score
            })

        return results

    finally:
        conn.close()


def _format_keyword_hits(
    fts_results: List[Dict[str, Any]],
    side: List[dict],
    chunk_text: Dict[str, str],
    k: int) -> List[SearchHit]:
    """Format FTS5 results into SearchHit objects"""
    out: List[SearchHit] = []

    # Create lookup map for metadata by chunk_id
    meta_lookup = {item['chunk_id']: item for item in side}

    for rank, result in enumerate(fts_results[:k], start=1):
        chunk_id = result['chunk_id']
        meta     = meta_lookup.get(chunk_id, {})

        out.append(
            SearchHit(
                rank=rank,
                score=result['fts_score'],
                chunk_id=chunk_id,
                doc_id=meta.get('doc_id'),
                page=meta.get('page'),
                title=meta.get('title'),
                text=result.get('text')  # FTS5 already returns the text
            )
        )

    return out


def _reciprocal_rank_fusion(
    semantic_hits: List[SearchHit],
    keyword_hits: List[SearchHit],
    k: int,
    rrf_constant: int = 60) -> List[SearchHit]:
    """
    Combine semantic and keyword search results using Reciprocal Rank Fusion (RRF).
    RRF score for item = sum(1 / (rank + rrf_constant)) across all result sets
    """
    rrf_scores = {}
    chunk_metadata = {}  # Store metadata for final results

    # Process semantic search results (rank-based scoring)
    for rank, hit in enumerate(semantic_hits, start=1):
        chunk_id             = hit.chunk_id
        rrf_score            = 1.0 / (rank + rrf_constant)
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + rrf_score

        # Store metadata (use semantic hit data as primary)
        if chunk_id not in chunk_metadata:
            chunk_metadata[chunk_id] = hit

    # Process keyword search results (rank-based scoring)
    for rank, hit in enumerate(keyword_hits, start=1):
        chunk_id             = hit.chunk_id
        rrf_score            = 1.0 / (rank + rrf_constant)
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + rrf_score

        # Store metadata if not already stored
        if chunk_id not in chunk_metadata:
            chunk_metadata[chunk_id] = hit

    # Sort by RRF score and create final results
    sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    final_hits = []
    for final_rank, (chunk_id, rrf_score) in enumerate(sorted_chunks[:k], start=1):
        original_hit = chunk_metadata[chunk_id]

        # Create new SearchHit with RRF score
        hybrid_hit = SearchHit(
            rank=final_rank,
            score=rrf_score,  # RRF combined score
            chunk_id=chunk_id,
            doc_id=original_hit.doc_id,
            page=original_hit.page,
            title=original_hit.title,
            text=original_hit.text
        )
        final_hits.append(hybrid_hit)

    return final_hits


def _bm25_rerank(
    query: str,
    hybrid_hits: List[SearchHit],
    bm25_index,
    bm25_chunk_ids: List[str],
    k: int,
    alpha: float = 0.7) -> List[SearchHit]:
    """
    Re-rank hybrid search results using BM25 scores.
    Combines RRF score with BM25 score using weighted sum.
    
    Args:
        query: Search query
        hybrid_hits: Results from RRF fusion of FAISS + FTS5
        bm25_index: Pre-built BM25 index
        bm25_chunk_ids: List of chunk_ids corresponding to BM25 index positions
        k: Number of final results to return
        alpha: Weight for RRF score (1-alpha for BM25 score)
    """
    if not bm25_index or not hybrid_hits:
        return hybrid_hits[:k]
    
    # Get BM25 scores for the query
    tokenized_query = query.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query)
    
    # Create lookup from chunk_id to BM25 score
    bm25_score_lookup = {}
    for i, chunk_id in enumerate(bm25_chunk_ids):
        bm25_score_lookup[chunk_id] = float(bm25_scores[i])
    
    # Normalize RRF scores to 0-1 range for fair combination
    if hybrid_hits:
        max_rrf_score = max(hit.score for hit in hybrid_hits)
        min_rrf_score = min(hit.score for hit in hybrid_hits)
        rrf_range = max_rrf_score - min_rrf_score if max_rrf_score > min_rrf_score else 1.0
    else:
        rrf_range = 1.0
        min_rrf_score = 0.0
    
    # Normalize BM25 scores to 0-1 range
    if bm25_scores.size > 0:
        max_bm25_score = float(bm25_scores.max())
        min_bm25_score = float(bm25_scores.min())
        bm25_range = max_bm25_score - min_bm25_score if max_bm25_score > min_bm25_score else 1.0
    else:
        max_bm25_score = min_bm25_score = bm25_range = 1.0
    
    # Combine scores and re-rank
    reranked_hits = []
    for hit in hybrid_hits:
        # Normalize RRF score
        normalized_rrf = (hit.score - min_rrf_score) / rrf_range
        
        # Get and normalize BM25 score
        bm25_score = bm25_score_lookup.get(hit.chunk_id, 0.0)
        normalized_bm25 = (bm25_score - min_bm25_score) / bm25_range if bm25_range > 0 else 0.0
        
        # Weighted combination
        combined_score = alpha * normalized_rrf + (1 - alpha) * normalized_bm25
        
        # Create new SearchHit with combined score
        reranked_hit = SearchHit(
            rank=0,  # Will be set after sorting
            score=combined_score,
            chunk_id=hit.chunk_id,
            doc_id=hit.doc_id,
            page=hit.page,
            title=hit.title,
            text=hit.text
        )
        reranked_hits.append(reranked_hit)
    
    # Sort by combined score and assign final ranks
    reranked_hits.sort(key=lambda x: x.score, reverse=True)
    for rank, hit in enumerate(reranked_hits[:k], start=1):
        hit.rank = rank
    
    return reranked_hits[:k]


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


@app.post("/keyword_search", response_model=KeywordSearchResponse)
def keyword_search(req: KeywordSearchRequest) -> KeywordSearchResponse:
    """
    Keyword-based search using FTS5.
    - Search chunks using SQLite FTS5 full-text search
    - Return top-k results ranked by FTS5 relevance score
    """
    state = cast(Any, getattr(app, "state"))

    if not state.fts5_available:
        raise HTTPException(status_code=503, detail="FTS5 search not available. Check your database setup.")

    # Perform FTS5 keyword search
    fts_results = _keyword_search(req.query, req.k * 2)  # Get extra results for filtering

    # Format results into SearchHit objects
    hits = _format_keyword_hits(fts_results, state.side, state.chunk_text, req.k)

    # Apply minimum score filter
    hits = [h for h in hits if h.score >= req.min_score][:req.k]

    return KeywordSearchResponse(query=req.query, k=req.k, hits=hits)


@app.post("/hybrid_search", response_model=HybridSearchResponse)
def hybrid_search(req: HybridSearchRequest) -> HybridSearchResponse:
    """
    Hybrid search combining semantic (FAISS) and keyword (FTS5) search using RRF,
    then re-ranked with BM25.
    - Performs both semantic and keyword search independently
    - Combines results using Reciprocal Rank Fusion (RRF)
    - Re-ranks the combined results using BM25 scores
    - Returns top-k results ranked by combined RRF + BM25 score
    """
    state = cast(Any, getattr(app, "state"))

    # Check if both search methods are available
    if state.index is None or state.model is None:
        raise HTTPException(status_code=503, detail="Semantic search not available.")

    if not state.fts5_available:
        raise HTTPException(status_code=503, detail="Keyword search not available.")

    # 1. Perform semantic search (reuse existing logic)
    qv = _embed_query(state.model, req.query)
    scores, ids = state.index.search(qv, req.semantic_k)
    semantic_hits = _format_hits(ids, scores, state.side, req.semantic_k, state.chunk_text)

    # Apply semantic minimum score filter
    semantic_hits = [h for h in semantic_hits if h.score >= req.min_score]

    # 2. Perform keyword search (reuse existing logic)
    fts_results = _keyword_search(req.query, req.keyword_k)
    keyword_hits = _format_keyword_hits(fts_results, state.side, state.chunk_text, req.keyword_k)

    # 3. Combine using Reciprocal Rank Fusion
    hybrid_hits = _reciprocal_rank_fusion(
        semantic_hits,
        keyword_hits,
        req.k * 2,  # Get more results for BM25 re-ranking
        req.rrf_constant
    )

    # 4. Re-rank with BM25 if available
    if state.bm25_index and hybrid_hits:
        final_hits = _bm25_rerank(
            req.query,
            hybrid_hits,
            state.bm25_index,
            state.bm25_chunk_ids,
            req.k,
            alpha=0.7  # 70% RRF score, 30% BM25 score
        )
        method = "hybrid_rrf_bm25_rerank"
    else:
        final_hits = hybrid_hits[:req.k]
        method = "hybrid_rrf"

    return HybridSearchResponse(
        query=req.query,
        k=req.k,
        hits=final_hits,
        method=method
    )


# -------------------------
# Local dev entry point
# -------------------------
if __name__ == "__main__":
    # Note: 'reload=True' is handy during development.
    uvicorn.run("search_api:app", host="127.0.0.1", port=8000, reload=True)
