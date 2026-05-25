import os
from typing import List, Dict, Any
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import Field, BaseModel

# --- Configuration ---
ES_HOST = "http://localhost:9200"
INDEX_NAME = "rag-hybrid-index"
EMBEDDING_MODEL_NAME = 'clip-ViT-B-32'
RERANKER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'


class HybridRetriever(BaseRetriever):
    """
    A LangChain-compatible retriever that performs hybrid search using Elasticsearch
    (BM25 + k-NN) and then reranks the results with a Cross-Encoder.
    """
    # 1. Declare fields at the class level for Pydantic
    es_client: Elasticsearch = Field(..., exclude=True)
    index_name: str
    embedding_model: SentenceTransformer = Field(..., exclude=True)
    reranker: CrossEncoder = Field(..., exclude=True)
    top_k: int

    # 2. Add Pydantic config to allow arbitrary types
    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        The core retrieval logic for LangChain's BaseRetriever.
        (This method remains unchanged)
        """
        # ... (your existing logic for _get_relevant_documents)
        # 1. Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()

        # 2. Perform k-NN (vector) search
        knn_query = {
            "field": "embedding",
            "query_vector": query_embedding,
            "k": 20,
            "num_candidates": 100,
        }
        knn_hits = self.es_client.search(
            index=self.index_name, knn=knn_query, source=True
        )["hits"]["hits"]

        # 3. Perform BM25 (keyword) search
        bm25_query = {
            "match": {
                "content": {"query": query, "operator": "and"}
            }
        }
        bm25_hits = self.es_client.search(
            index=self.index_name, query=bm25_query, source=True, size=20
        )["hits"]["hits"]

        # 4. Combine and deduplicate results
        combined_hits: Dict[str, Dict[str, Any]] = {}
        for hit in bm25_hits + knn_hits:
            combined_hits[hit["_id"]] = hit

        # 5. Rerank the combined results
        passages = [
            hit["_source"]["content"]
            for hit in combined_hits.values()
            if hit["_source"].get("content")
        ]
        passage_map = {
            hit["_source"]["content"]: hit for hit in combined_hits.values() if hit["_source"].get("content")
        }

        if not passages:
            return []

        rerank_pairs = [(query, p) for p in passages]
        rerank_scores = self.reranker.predict(rerank_pairs)

        reranked_results = [
            (score, passage_map[passage]) for score, passage in zip(rerank_scores, passages)
        ]
        reranked_results.sort(key=lambda x: x[0], reverse=True)

        # 6. Format results into LangChain Document objects
        final_documents = []
        for score, hit in reranked_results[: self.top_k]:
            source_data = hit["_source"]
            metadata = {
                "doc_id": source_data.get("doc_id"),
                "type": source_data.get("type"),
                "chunk_type": source_data.get("chunk_type"),
                "image_path": source_data.get("image_path"),
                "rerank_score": float(score),
                "original_es_score": hit["_score"],
            }
            metadata = {k: v for k, v in metadata.items() if v is not None}
            
            if source_data.get("type") == "image":
                # Return image path and optional caption if desired
                doc = Document(
                    page_content="[[IMAGE]]",
                    metadata={**metadata, "image_path": source_data.get("image_path")}
                )
            else:
                doc = Document(
                    page_content=source_data.get("content", ""), metadata=metadata
                )
            final_documents.append(doc)

        return final_documents

# You no longer need the __init__ method, as Pydantic handles initialization automatically
# The rest of your file (get_retriever() and the __main__ block) can remain as is.

def get_retriever() -> HybridRetriever:
    """Factory function to create an instance of the HybridRetriever."""
    print("Initializing retriever components...")
    es_client = Elasticsearch(ES_HOST)
    if not es_client.ping():
        raise ConnectionError(f"Could not connect to Elasticsearch at {ES_HOST}")

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    reranker = CrossEncoder(RERANKER_MODEL_NAME)
    
    # Pydantic will now correctly initialize this class
    return HybridRetriever(
        es_client=es_client,
        index_name=INDEX_NAME,
        embedding_model=embedding_model,
        reranker=reranker,
        top_k=5,
    )


if __name__ == "__main__":
    try:
        retriever = get_retriever()
        query_text = "Telescopes"

        print(f"\n--- Retrieving documents for query: '{query_text}' ---")
        results = retriever.get_relevant_documents(query_text)

        if not results:
            print("No relevant documents found.")
        else:
            print(f"\n--- Top {len(results)} reranked results ---")
            for idx, doc in enumerate(results, 1):
                print(f"\nResult {idx}:")
                print(f"  Rerank Score: {doc.metadata.get('rerank_score'):.4f}")
                print(f"  Content: {doc.page_content[:300]}...")
                print(f"  Metadata: {doc.metadata}")

    except (ConnectionError, ImportError) as e:
        print(f"\nERROR: Could not run retriever example. {e}")
        print("Please ensure Elasticsearch is running and all dependencies are installed (`pip install langchain elasticsearch sentence-transformers pydantic`).")

