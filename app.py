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
    question: Optional[str] = None
    top_k: int = 5

    @property
    def query(self):
        return self.question


@app.get("/")
def root():
    return {"message": "TDS Virtual TA is live!"}


@app.post("/")
async def ask_question(request: QueryRequest):
    if not request.query:
        return {"error": "Query text is required"}

    query_embedding = model.encode(request.query, normalize_embeddings=True)
    scores = np.dot(post_embeddings, query_embedding)
    top_indices = np.argsort(scores)[::-1][:request.top_k]

    results = [posts[i] for i in top_indices]
    answer = " ".join(post.get("text", "") for post in results if post.get("text"))[:1000]

    links = [
        {"url": post.get("url", ""), "text": post.get("title", "")}
        for post in results if post.get("url")
    ]
    while len(links) < 3:
        links.append({"url": "", "text": ""})

    return {
        "answer": answer or "Sorry, no relevant content found.",
        "links": links[:3]
    }


@app.post("/search")
async def search(request: QueryRequest):
    if not request.query:
        return {"error": "Query text is required"}

    query_embedding = model.encode(request.query, normalize_embeddings=True)
    scores = np.dot(post_embeddings, query_embedding)
    top_indices = np.argsort(scores)[::-1][:request.top_k]

    return {
        "results": [
            {
                "title": posts[i].get("title", ""),
                "url": posts[i].get("url", ""),
                "score": float(scores[i]),
                "excerpt": posts[i].get("text", "")[:300]
            }
            for i in top_indices
        ]
    }
