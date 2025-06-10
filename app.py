from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Load reduced JSON data (small subset of posts only)
with open("tds_discourse_posts_small.json", "r") as f:
    posts = json.load(f)

# Load a lightweight embedding model (Render free-tier safe)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load precomputed embeddings
post_embeddings = np.array([post["embedding"] for post in posts])

# Initialize FastAPI app
app = FastAPI()

# Allow CORS for any origin (frontend compatibility)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request body model
class QueryRequest(BaseModel):
    query: Optional[str] = None
    top_k: int = 5

# Health check route
@app.get("/")
def read_root():
    return {"message": "TDS Virtual TA is running!"}

# Search endpoint
@app.post("/search")
async def search(request: QueryRequest):
    if not request.query:
        return {"error": "Query text is required"}

    # Encode query and compute similarity
    query_embedding = model.encode(request.query, normalize_embeddings=True)
    scores = np.dot(post_embeddings, query_embedding)
    top_indices = np.argsort(scores)[::-1][:request.top_k]

    # Prepare response
    results = []
    for i in top_indices:
        post = posts[i]
        results.append({
            "title": post.get("title", ""),
            "url": post.get("url", ""),
            "score": float(scores[i]),
            "excerpt": post.get("excerpt", "")[:300]
        })

    return {"results": results}
