import traceback
import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import torch

app = FastAPI()

# Load the compressed posts
file_path = os.path.join(os.path.dirname(__file__), "tds_discourse_posts_compressed.json")
with open(file_path, "r", encoding="utf-8") as f:
    posts = json.load(f)

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")
model.to(torch.device("cpu"))

class QuestionRequest(BaseModel):
    question: str

@app.post("/api/")
async def get_answer(request: QuestionRequest):
    question = request.question
    try:
        question_embedding = model.encode(question, convert_to_tensor=True, device="cpu")

        sims = []
        for post in posts:
            emb = post.get("embedding")
            if emb is None:
                continue
            emb_tensor = torch.tensor(emb, dtype=torch.float32, device="cpu")
            sim = util.cos_sim(question_embedding, emb_tensor)[0][0].item()
            sims.append((sim, post))

        top_posts = sorted(sims, key=lambda x: x[0], reverse=True)[:3]

        contents = [post.get("text", "") for _, post in top_posts if post.get("text")]
        answer = " ".join(contents) if contents else "Sorry, relevant content not found."

        links = []
        for _, post in top_posts:
            links.extend(post.get("links", []))
        while len(links) < 3:
            links.append("")

        return {"answer": answer, "links": links[:3]}

    except Exception as e:
        error_message = f"Exception: {str(e)}\nTraceback:\n{traceback.format_exc()}"
        print(error_message)
        return {"answer": "Internal error occurred.", "links": [], "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
