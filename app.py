from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json
import numpy as np
import base64
import torch
from sentence_transformers import SentenceTransformer
from PIL import Image
from io import BytesIO
import os

# Load embeddings and posts from the smaller JSON file
with open("tds_discourse_posts_small.json", "r") as f:
    posts = json.load(f)

post_embeddings = np.array([post["embedding"] for post in posts])

# Load the CLIP model
model = SentenceTransformer("clip-ViT-B-32")

# FastAPI app setup
app = FastAPI()

# Enable CORS for local development or frontend testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request and response models
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
            "excerpt": post.get("excerpt", "")[:300]  # Limit excerpt size
        })

    return {"results": results}

@app.post("/search-image")
async def search_image(file: UploadFile = File(...), top_k: int = 5):
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_embedding = model.encode(image, normalize_embeddings=True)
    scores = np.dot(post_embeddings, image_embedding)
    top_indices = np.argsort(scores)[::-1][:top_k]

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
