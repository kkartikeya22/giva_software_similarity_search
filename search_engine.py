import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

class DocumentSearch:
    def __init__(self, file_path="data/documents.txt"):
        self.file_path = file_path
        self.documents = self.load_documents()
        self.index, self.embeddings = self.create_faiss_index()
    
    def load_documents(self):
        """Load documents from a text file"""
        with open(self.file_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines()]
    
    def create_faiss_index(self):
        """Convert documents to embeddings & store in FAISS"""
        embeddings = model.encode(self.documents, convert_to_numpy=True)
        index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance
        index.add(embeddings)
        return index, embeddings
    
    def search(self, query, top_k=5, metric="cosine"):
        """Find top-k similar documents for a query"""
        query_embedding = model.encode([query], convert_to_numpy=True)
        
        if metric == "cosine":
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(query_embedding)
            faiss.normalize_L2(self.embeddings)
        
        distances, indices = self.index.search(query_embedding, top_k)
        results = [{"document": self.documents[idx], "score": float(1 - d if metric == "cosine" else d)}
                   for d, idx in zip(distances[0], indices[0])]
        return results
