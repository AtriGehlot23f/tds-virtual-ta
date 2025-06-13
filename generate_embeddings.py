import json
from sentence_transformers import SentenceTransformer
import numpy as np


model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(text):
    
    embedding = model.encode(text)
    return embedding

def add_chunks(course_data):
    all_chunks = []
    for entry in course_data:
        
        
        if isinstance(entry, dict):
            text = entry.get("text") or entry.get("content") or ""
            if text.strip():
                embedding = embed_text(text)
                all_chunks.append({
                    "id": entry.get("id"),
                    "text": text,
                    "embedding": embedding.tolist()
                })
    return all_chunks

def main():
    
    with open("tds_discourse_posts.json", "r") as f:
        course_data = json.load(f)

    print(f"Loaded {len(course_data)} posts for embedding.")

    embedded_data = add_chunks(course_data)

    
    with open("tds_discourse_posts_with_embeddings.json", "w") as f:
        json.dump(embedded_data, f, indent=2)

    print(f"Saved {len(embedded_data)} embedded posts.")

if __name__ == "__main__":
    main()
