from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List


with open("tds_discourse_posts_small.json", "r") as f:
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
    question: str


class QueryResponse(BaseModel):
    answer: str
    links: List[str]


@app.get("/")
def read_root():
    return {"message": "TDS Virtual TA is running!"}


@app.post("/api/", response_model=QueryResponse)
async def get_answer(request: QueryRequest):
    question = request.question

    
    question_embedding = model.encode(question, normalize_embeddings=True)

    
    scores = np.dot(post_embeddings, question_embedding)
    top_indices = np.argsort(scores)[::-1][:3]

    
    top_posts = [posts[i] for i in top_indices]
    answer_parts = [p.get("excerpt", "") for p in top_posts]
    answer = " ".join(answer_parts) if answer_parts else "Sorry, no relevant posts found."

    links = [p.get("url", "") for p in top_posts]
    while len(links) < 3:
        links.append("")

    return {"answer": answer, "links": links}
