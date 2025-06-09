import traceback
import os
import json
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util

app = FastAPI()

# Load JSON data
with open("tds_discourse_posts_with_embeddings.json", "r", encoding="utf-8") as f:
    posts = json.load(f)

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")
model.to(torch.device("cpu"))

# Input schema
class QuestionRequest(BaseModel):
    question: str

@app.post("/api/")
async def get_answer(request: QuestionRequest):
    try:
        question = request.question
        question_embedding = model.encode(question, convert_to_tensor=True, device="cpu")

        similarities = []
        for post in posts:
            emb = post.get("embedding")
            if emb:
                emb_tensor = torch.tensor(emb, device="cpu")
                sim = util.cos_sim(question_embedding, emb_tensor)[0][0].item()
                similarities.append((sim, post))

        # Get top 3 similar posts
        top_posts = sorted(similarities, key=lambda x: x[0], reverse=True)[:3]

        # Answer: join their text
        contents = [p.get("text", "") for _, p in top_posts if p.get("text")]
        answer = " ".join(contents) if contents else "Sorry, relevant content not found."

        # Collect up to 3 links
        links = []
        for _, post in top_posts:
            links.extend(post.get("links", []))
        while len(links) < 3:
            links.append("")

        return {"answer": answer, "links": links[:3]}

    except Exception as e:
        print("Exception:", e)
        print(traceback.format_exc())
        return {"answer": "Internal error occurred.", "links": [], "error": str(e)}

# For local/dev testing (Render ignores this block)
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
