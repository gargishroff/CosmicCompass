import os
import json
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from PIL import Image

# --- Configuration ---
# These paths should point to the output directories from your processing script.

CHUNK_DIR = os.path.join(os.getcwd(), "outputs/chunks")
IMAGE_DIR = os.path.join(os.getcwd(), "outputs/images")
METADATA_DIR = os.path.join(os.getcwd(), "outputs/metadata")

# --- Elasticsearch & Model Configuration ---
ES_HOST = "http://localhost:9200"
INDEX_NAME = "rag-hybrid-index-v2"
# Using a multimodal model allows encoding both text and images.
MODEL_NAME = 'clip-ViT-B-32'


class HybridIndexer:
    """
    Handles the creation of a hybrid search index in Elasticsearch and
    the indexing of text and image data based on metadata files.
    """

    def __init__(self, es_host, index_name, model_name):
        """
        Initializes the connection to Elasticsearch and loads the embedding model.
        """
        print("Connecting to Elasticsearch...")
        self.es_client = Elasticsearch(es_host)
        if not self.es_client.ping():
            raise ConnectionError(f"Could not connect to Elasticsearch at {es_host}")

        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        # Get the embedding dimension from the model configuration
        self.embedding_dim = 512
        self.index_name = index_name
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")

    def create_index_if_not_exists(self):
        """
        Creates the Elasticsearch index with a mapping optimized for hybrid search.
        The mapping includes a 'dense_vector' field for semantic search and a
        'text' field for keyword (BM25) search.
        """
        if not self.es_client.indices.exists(index=self.index_name):
            print(f"Creating index '{self.index_name}' with hybrid mapping...")
            mapping = {
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "type": {"type": "keyword"}, # 'text' or 'image'
                    "chunk_type": {"type": "keyword"}, # 'parent' or 'child'
                    # The 'content' field is analyzed for full-text search (BM25)
                    "content": {"type": "text"},
                    # The 'embedding' field stores the vector for k-NN search
                    "embedding": {
                        "type": "dense_vector",
                        "dims": self.embedding_dim,
                        "index": True,
                        "similarity": "cosine" # Cosine similarity is common for text embeddings
                    },
                    "image_path": {"type": "keyword"},
                }
            }
            self.es_client.indices.create(index=self.index_name, mappings=mapping)
            print(f"Index '{self.index_name}' created successfully.")
        else:
            print(f"Index '{self.index_name}' already exists.")

    def _index_document(self, es_id, document):
        """Helper function to index a single document."""
        try:
            self.es_client.index(
                index=self.index_name,
                id=es_id,
                document=document
            )
        except Exception as e:
            print(f"Error indexing document {es_id}: {e}")

    def index_text_chunk(self, doc_id, chunk_filename, chunk_type):
        """
        Reads a text chunk, generates its embedding, and indexes it.
        """
        chunk_path = os.path.join(CHUNK_DIR, chunk_filename)
        if not os.path.exists(chunk_path):
            print(f"  -> Warning: Chunk file not found, skipping: {chunk_filename}")
            return

        with open(chunk_path, 'r', encoding='utf-8') as f:
            text_content = f.read()

        print(f"  - Indexing text chunk: {chunk_filename}")
        text_embedding = self.model.encode(text_content).tolist()

        text_doc = {
            "embedding": text_embedding,
            "doc_id": doc_id,
            "type": "text",
            "chunk_type": chunk_type,
            "content": text_content
        }
        # Create a unique ID for the Elasticsearch document
        chunk_es_id = f"text_{doc_id}_{chunk_filename}"
        self._index_document(chunk_es_id, text_doc)

    def index_image(self, doc_id, img_filename):
        """
        Opens an image, generates its embedding, and indexes it.
        """
        image_path = os.path.join(IMAGE_DIR, img_filename)
        if not os.path.exists(image_path):
            print(f"  -> Warning: Image file not found, skipping: {img_filename}")
            return

        try:
            print(f"  - Indexing image: {img_filename}")
            # Ensure image is in RGB format, which the CLIP model expects
            image = Image.open(image_path).convert("RGB")
            image_embedding = self.model.encode(image).tolist()
            print(image_path)
            image_doc = {
                "embedding": image_embedding,
                "doc_id": doc_id,
                "type": "image",
                "image_path": image_path,
                # 'content' field is null for images
                "content": None
            }
            image_es_id = f"image_{doc_id}_{img_filename}"
            self._index_document(image_es_id, image_doc)
        except Exception as e:
            print(f"  -> ERROR: Could not process image {img_filename}: {e}")

    def run_indexing(self):
        """
        Main function to scan the metadata directory and index all assets.
        """
        metadata_files = [f for f in os.listdir(METADATA_DIR) if f.endswith('.json')]

        if not metadata_files:
            print(f"No metadata files found in '{METADATA_DIR}'. Run the document processing script first.")
            return

        for meta_file in metadata_files:
            meta_path = os.path.join(METADATA_DIR, meta_file)
            with open(meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            doc_id = metadata['doc_id']
            print(f"\n--- Indexing assets for document: {doc_id} ---")

            # 1. Index all text chunks (child and parent)
            text_assets = metadata['assets'].get('child_chunks', []) + metadata['assets'].get('parent_chunks', [])
            chunk_types = (["child"] * len(metadata['assets'].get('child_chunks', [])) +
                           ["parent"] * len(metadata['assets'].get('parent_chunks', [])))

            for chunk_file, chunk_type in zip(text_assets, chunk_types):
                self.index_text_chunk(doc_id, chunk_file, chunk_type)

            # 2. Index all images
            for img_file in metadata['assets'].get('images', []):
                self.index_image(doc_id, img_file)

        print("\n--- Elasticsearch indexing process complete! ---")
        self.es_client.indices.refresh(index=self.index_name)
        count = self.es_client.count(index=self.index_name).get('count', 0)
        print(f"Total documents in index '{self.index_name}': {count}")


if __name__ == '__main__':
    # Initialize and run the indexer
    indexer = HybridIndexer(es_host=ES_HOST, index_name=INDEX_NAME, model_name=MODEL_NAME)

    # 1. Set up the index mapping
    indexer.create_index_if_not_exists()

    # 2. Run the indexing process
    indexer.run_indexing()

