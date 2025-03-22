import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

# Enable Debug Logging
logging.basicConfig(level=logging.DEBUG)

# Load sentence transformer model
try:
    logging.debug("Loading sentence transformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    logging.debug("Model loaded successfully!")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise

class DocumentSearch:
    def __init__(self, file_path="data/documents.txt"):
        try:
            logging.debug(f"Loading documents from {file_path}...")
            self.file_path = file_path
            self.documents = self.load_documents()
            if not self.documents:
                logging.warning("No documents found! Using a default dataset.")
                self.documents = ["This is a default document for testing."]

            logging.debug(f"Loaded {len(self.documents)} documents.")

            # ✅ Ensure `self.embeddings` is always initialized
            self.embeddings = None
            self.index, self.embeddings = self.create_faiss_index()

        except Exception as e:
            logging.error(f"Error initializing DocumentSearch: {e}")
            raise

    def load_documents(self):
        """Load documents from a text file"""
        try:
            if not os.path.exists(self.file_path):
                logging.warning(f"File {self.file_path} not found! Using default documents.")
                return ["AI is transforming the world.", "Machine learning is a subset of AI."]
            
            with open(self.file_path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f.readlines() if line.strip()]
        except Exception as e:
            logging.error(f"Error loading documents: {e}")
            return ["AI is transforming the world.", "Machine learning is a subset of AI."]

    def create_faiss_index(self):
    """Convert documents to embeddings & store in FAISS"""
    try:
        if not self.documents:
            logging.error("No documents found! FAISS index cannot be created.")
            return None, None

        logging.debug("Generating embeddings for documents...")
        embeddings = model.encode(self.documents, convert_to_numpy=True)

        if embeddings.shape[0] == 0:
            logging.error("Embeddings could not be generated! FAISS index cannot be created.")
            return None, None

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        logging.debug("FAISS index created successfully!")
        return index, embeddings
    except Exception as e:
        logging.error(f"Error creating FAISS index: {e}")
        return None, None


    def search(self, query, top_k=5, metric="cosine"):
        """Find top-k similar documents for a query"""
        try:
            if self.embeddings is None:
                logging.error("FAISS index is not initialized! Ensure documents.txt is valid.")
                return {"error": "FAISS index not initialized. Check document loading."}

            logging.debug(f"Searching for query: {query} with metric: {metric}")
            query_embedding = model.encode([query], convert_to_numpy=True)

            if metric == "cosine":
                faiss.normalize_L2(query_embedding)
                faiss.normalize_L2(self.embeddings)

            distances, indices = self.index.search(query_embedding, top_k)
            results = [{"document": self.documents[idx], "score": float(1 - d if metric == "cosine" else d)}
                    for d, idx in zip(distances[0], indices[0])]
            return results
        except Exception as e:
            logging.error(f"Error in search function: {e}")
            return {"error": "Search failed due to internal error."}
