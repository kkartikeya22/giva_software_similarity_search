import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model once (lazy loading)
model = None
faiss_index = None
documents = []

def load_model():
    """Load model and FAISS index only once to optimize memory usage."""
    global model, faiss_index, documents

    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")

    # Load documents from an environment variable or default file
    file_path = os.environ.get("DOCUMENTS_PATH", "data/documents.txt")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            documents = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print(f"⚠️ Warning: {file_path} not found! Using default dataset.")
        documents = ["This is a sample document.", "AI is transforming the world."]

    # Create FAISS index
    embeddings = model.encode(documents, convert_to_numpy=True)
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(embeddings)

class DocumentSearch:
    def __init__(self):
        """Ensure FAISS and model are loaded before using."""
        if model is None or faiss_index is None:
            load_model()

    def search(self, query, top_k=5, metric="cosine"):
        """Find top-k similar documents for a query."""
        query_embedding = model.encode([query], convert_to_numpy=True)

        if metric == "cosine":
            faiss.normalize_L2(query_embedding)
            faiss.normalize_L2(self.embeddings)

        distances, indices = faiss_index.search(query_embedding, top_k)
        results = [{"document": documents[idx], "score": float(1 - d if metric == "cosine" else d)}
                   for d, idx in zip(distances[0], indices[0])]
        return results
