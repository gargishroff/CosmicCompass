import functools
import os
from typing import List, Dict, Any
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import Field
from langchain.tools import tool

# --- Configuration ---
ES_HOST = "http://localhost:9200"
INDEX_NAME = "rag-hybrid-index-v2"
EMBEDDING_MODEL_NAME = 'clip-ViT-B-32'
RERANKER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'


class HybridRetriever(BaseRetriever):
    """
    A LangChain-compatible retriever that performs hybrid search,
    separately handling text (for reranking) and images.
    """
    es_client: Elasticsearch = Field(..., exclude=True)
    index_name: str
    embedding_model: SentenceTransformer = Field(..., exclude=True)
    reranker: CrossEncoder = Field(..., exclude=True)
    top_k: int  # This will now apply to the top_k *text* documents

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, min_images: int = 0) -> List[Document]:
        """
        The core retrieval logic, now supporting multi-modal results
        and a minimum image requirement.
        """
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

        # 4. Combine and deduplicate
        combined_hits: Dict[str, Dict[str, Any]] = {}
        for hit in bm25_hits + knn_hits:
            combined_hits[hit["_id"]] = hit

        # 5. --- NEW LOGIC: Separate Text and Images ---
        text_hits_to_rerank = []
        image_hits_to_keep = []
        for hit in combined_hits.values():
            source = hit["_source"]
            if source.get("type") == "image":
                image_hits_to_keep.append(hit)
            elif source.get("content"):
                text_hits_to_rerank.append(hit)
        
        # 6. --- NEW LOGIC: Handle min_images ---
        if len(image_hits_to_keep) < min_images:
            # Not enough images found, do a dedicated image-only k-NN search
            image_knn_query = {
                "field": "embedding",
                "query_vector": query_embedding,
                "k": min_images,
                "num_candidates": 50,
                "filter": {
                    "term": { "type": "image" } # Filter for type: image
                }
            }
            more_image_hits = self.es_client.search(
                index=self.index_name, knn=image_knn_query, source=True
            )["hits"]["hits"]

            # Add new images, avoiding duplicates
            current_image_ids = {h["_id"] for h in image_hits_to_keep}
            for hit in more_image_hits:
                if hit["_id"] not in current_image_ids:
                    image_hits_to_keep.append(hit)

        # 7. Rerank *only* the text results
        reranked_text_results = []
        if text_hits_to_rerank:
            passages = [hit["_source"]["content"] for hit in text_hits_to_rerank]
            passage_map = {hit["_source"]["content"]: hit for hit in text_hits_to_rerank}
            
            rerank_pairs = [(query, p) for p in passages]
            rerank_scores = self.reranker.predict(rerank_pairs)
            
            reranked_text_results = [
                (score, passage_map[passage]) for score, passage in zip(rerank_scores, passages)
            ]
            reranked_text_results.sort(key=lambda x: x[0], reverse=True)

        # 8. Format the final list
        final_documents = []
        
        # Add top_k reranked text documents
        for score, hit in reranked_text_results[:self.top_k]:
            source_data = hit["_source"]
            metadata = {
                "doc_id": source_data.get("doc_id"),
                "type": "text",
                "chunk_type": source_data.get("chunk_type"),
                "rerank_score": float(score),
                "original_es_score": hit["_score"],
            }
            metadata = {k: v for k, v in metadata.items() if v is not None}
            doc = Document(page_content=source_data.get("content", ""), metadata=metadata)
            final_documents.append(doc)
            
        # Add all found/required image documents
        for hit in image_hits_to_keep:
            source_data = hit["_source"]
            metadata = {
                "doc_id": source_data.get("doc_id"),
                "type": "image",
                "image_path": source_data.get("image_path"),
                "original_es_score": hit["_score"],
            }
            metadata = {k: v for k, v in metadata.items() if v is not None}
            doc = Document(
                page_content=f"[[Image retrieved from corpus: {os.path.basename(source_data.get('image_path', ''))}]]",
                metadata=metadata
            )
            final_documents.append(doc)

        return final_documents


@functools.lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    """Factory function to create a cached instance of the HybridRetriever."""
    print("Initializing retriever components (this should only happen once)...")
    es_client = Elasticsearch(ES_HOST)
    if not es_client.ping():
        raise ConnectionError(f"Could not connect to Elasticsearch at {ES_HOST}")

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    reranker = CrossEncoder(RERANKER_MODEL_NAME)
    
    return HybridRetriever(
        es_client=es_client,
        index_name=INDEX_NAME,
        embedding_model=embedding_model,
        reranker=reranker,
        top_k=5,  # This will be the top_k *text* results
    )


@tool
def rag_retriever_tool(query: str, min_images: int = 0) -> List[Document]:
    """
    Tool wrapper for the HybridRetriever.
    
    Args:
        query (str): The search query for text and images.
        min_images (int, optional): The minimum number of images
                                     to retrieve. Defaults to 0.
    """
    retriever = get_retriever()
    return retriever._get_relevant_documents(query, min_images=min_images)