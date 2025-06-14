import json
import numpy as np
from sentence_transformers import SentenceTransformer


with open("tds_discourse_posts_with_embeddings.json", "r") as f:
    embedded_posts = json.load(f)


embeddings = np.array([post["embedding"] for post in embedded_posts])


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search(query, top_k=5):
    """Search for the top_k most similar posts to the query."""
    query_embedding = model.encode(query)
    
    
    scores = [cosine_similarity(query_embedding, emb) for emb in embeddings]
    
    
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    
    results = []
    for idx in top_indices:
        post = embedded_posts[idx]
        score = scores[idx]
        results.append((post, score))
    return results

if __name__ == "__main__":
    user_query = input("Enter your question: ").strip()
    results = search(user_query, top_k=5)
    
    print("\nTop 5 results:\n")
    for post, score in results:
        print(f"Score: {score:.4f}")
        print(f"URL: {post.get('url', 'URL not available')}")
        # Show first 400 chars of text to keep output manageable
        print(f"Text snippet: {post.get('text', 'No text available')[:400]}...\n")
