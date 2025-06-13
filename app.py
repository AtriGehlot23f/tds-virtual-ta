from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import json
import numpy as np
from sentence_transformers import SentenceTransformer


with open("tds_discourse_posts_with_embeddings.json", "r") as f:
    posts = json.load(f)


model = SentenceTransformer("all-MiniLM-L6-v2")


post_embeddings = np.array([post["embedding"] for post in posts])


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: Optional[str] = None
    top_k: int = 5


@app.get("/")
def read_root():
    return {"message": "TDS Virtual TA is running!"}


@app.post("/search")
async def search(request: QueryRequest):
    if not request.query:
        return {"error": "Query text is required"}

    
    query_embedding = model.encode(request.query, normalize_embeddings=True)
    scores = np.dot(post_embeddings, query_embedding)
    top_indices = np.argsort(scores)[::-1][:request.top_k]

    
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


@app.post("/")
async def root_post(request: QueryRequest):
    return await search(request)
