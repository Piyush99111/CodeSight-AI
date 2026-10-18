"""
CodeSight-AI: Retrieval-Augmented Generation (RAG) Pipeline
Handles ChromaDB vector storage, embedding generation, and grounded context injection.
"""

import chromadb
from sentence_transformers import SentenceTransformer

class CodeSightRAGEngine:
    def __init__(self, collection_name: str = "codesight_live_docs"):
        print("Initializing Embedding Model (BAAI/bge-small-en-v1.5)...")
        self.embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda" if torch.cuda.is_available() else "cpu")
        
        print("Initializing ChromaDB Vector Store...")
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def ingest_documents(self, documents: list):
        """
        Ingests technical documentation chunks into ChromaDB with embeddings.
        """
        texts = [doc["content"] for doc in documents]
        ids = [doc["id"] for doc in documents]
        metadatas = [{"title": doc["title"]} for doc in documents]

        embeddings = self.embed_model.encode(texts, normalize_embeddings=True).tolist()

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        print(f"Successfully indexed {len(documents)} document chunks into ChromaDB.")

    def retrieve_context(self, query: str, top_k: int = 1):
        """
        Retrieves the most semantically relevant documentation chunk for a given query.
        """
        query_vector = self.embed_model.encode([query], normalize_embeddings=True).tolist()
        results = self.collection.query(
            query_embeddings=query_vector,
            n_results=top_k
        )
        
        doc = results["documents"][0][0]
        title = results["metadatas"][0][0]["title"]
        distance = results["distances"][0][0]
        
        return doc, title, distance

if __name__ == "__main__":
    print("CodeSight-AI RAG module initialized.")