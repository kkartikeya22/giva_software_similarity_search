import os
from fastapi import FastAPI, Query
from search_engine import DocumentSearch
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Enable CORS (optional for frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this for security in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the search engine
search_engine = DocumentSearch()

@app.get("/")
def home():
    """Welcome message for the API."""
    return {"message": "Welcome to Document Similarity Search API!"}

@app.get("/health")
def health_check():
    """Health check endpoint to verify if API is running."""
    return {"status": "healthy"}

@app.get("/api/search")
async def search_documents(q: str = Query(..., title="Search Query"), metric: str = "cosine"):
    """
    Search for similar documents based on a query.
    Metrics: cosine | l2
    """
    if metric not in ["cosine", "l2"]:
        return {"error": "Invalid metric! Use 'cosine' or 'l2'."}
    
    results = await search_engine.search(q, top_k=5, metric=metric)
    return {"query": q, "results": results}

# Local Development vs. Deployment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Default to 8000 locally
    host = "127.0.0.1" if os.environ.get("RENDER") is None else "0.0.0.0"
    
    import hypercorn.asyncio
    from hypercorn.config import Config
    import asyncio

    config = Config()
    config.bind = [f"{host}:{port}"]  # Bind to local 127.0.0.1 or Render's 0.0.0.0

    asyncio.run(hypercorn.asyncio.serve(app, config))
