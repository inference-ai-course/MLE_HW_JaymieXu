from __future__ import annotations
from typing import Any, Dict, List, Optional

from rag2.rag_engine import RAGEngine

# ---------------------------
# Data classes (replacing Pydantic models)
# ---------------------------
class SearchHit:
    def __init__(self, rank: int, score: float, chunk_id: str,
                 doc_id: Optional[str] = None, page: Optional[int] = None,
                 title: Optional[str] = None, text: Optional[str] = None):
        self.rank = rank
        self.score = score
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.page = page
        self.title = title
        self.text = text


# ------------------------------------------------
# Search Class
# ------------------------------------------------
class Search:
    def __init__(self):
        try:
            self.rag_search = RAGEngine(collection_name="demo_collection")
            print("RAG Engine loaded.")
        except Exception as e:
            print(f"RAG Engine failed to load: {e}")
            self.rag_search = None
            
    
    def result_to_searchhit(self, llama_index_result : List[Dict[str, Any]]) -> List[SearchHit]:
        if not llama_index_result:
            return []

        search_hits = []
        for result in llama_index_result:
            hit = SearchHit(
                rank=result['rank'],
                score=result['score'],
                chunk_id='N/A',
                doc_id=result.get('doc_id', 'N/A'),
                page=None,
                title=result.get('metadata', {}).get('title', 'N/A'),
                text=result.get('text', 'N/A')
            )
            search_hits.append(hit)

        return search_hits


    def hybrid_search(self, query: str, k: int = 5) -> List[SearchHit]:
        return self.result_to_searchhit(self.rag_search.hybrid_search(query, k))
