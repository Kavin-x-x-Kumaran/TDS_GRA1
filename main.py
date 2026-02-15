import os
import time
import asyncio
import numpy as np
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List, Dict

# --- Configuration ---
AI_PIPE_TOKEN = os.getenv("AI_PIPE_TOKEN", "your_token_here")
AI_PIPE_URL = "https://aipipe.org/openrouter/v1/chat/completions"
MODEL_NAME = "openai/gpt-4o-mini"
EMBED_MODEL = "all-MiniLM-L6-v2"  # Fast, 384d, <200ms retrieval

app = FastAPI(title="SearchTech Two-Stage Pipeline")

# --- In-Memory Store ---
# Generating 125 mock abstracts for the demo
DOCS = [
    {"id": i, "text": f"Abstract {i}: Scientific discovery regarding AI in search engines focusing on {topic}."}
    for i, topic in enumerate(["vector databases", "re-ranking", "transformers", "LLMs", "latency"] * 25)
]
DOC_TEXTS = [d["text"] for d in DOCS]

# Initialize Model & Embeddings
model = SentenceTransformer(EMBED_MODEL)
doc_embeddings = model.encode(DOC_TEXTS, convert_to_numpy=True)
# Normalize for Cosine Similarity (Dot product of normalized vectors = Cosine Sim)
doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)

class SearchRequest(BaseModel):
    query: str
    rerank: bool = True

class SearchResponse(BaseModel):
    results: List[Dict]
    reranked: bool
    metrics: Dict

# --- Helper: AI PIPE Client ---
async def get_llm_score(client: httpx.AsyncClient, query: str, doc_text: str) -> float:
    prompt = f"Query: {query}\nDocument: {doc_text}\n\nRate the relevance of this document to the query on a scale of 0-10. Respond with only the number."
    try:
        response = await client.post(
            AI_PIPE_URL,
            headers={"Authorization": f"Bearer {AI_PIPE_TOKEN}"},
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0
            },
            timeout=5.0
        )
        res_json = response.json()
        score_raw = res_json['choices'][0]['message']['content'].strip()
        # Normalize 0-10 to 0-1
        return float(score_raw) / 10.0
    except Exception:
        return 0.0

# --- API Endpoint ---
@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    start_time = time.time()
    
    # --- STAGE 1: Vector Search ---
    query_embedding = model.encode([request.query], convert_to_numpy=True)
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    
    # Calculate Cosine Similarity
    similarities = np.dot(doc_embeddings, query_embedding.T).flatten()
    top_indices = np.argsort(similarities)[::-1][:8]
    
    retrieval_results = []
    for idx in top_indices:
        retrieval_results.append({
            "id": DOCS[idx]["id"],
            "text": DOCS[idx]["text"],
            "score": float(similarities[idx])
        })
    
    retrieval_time = (time.time() - start_time) * 1000
    
    # --- STAGE 2: LLM Re-ranking ---
    rerank_start = time.time()
    final_results = retrieval_results
    
    if request.rerank:
        async with httpx.AsyncClient() as client:
            tasks = [get_llm_score(client, request.query, res["text"]) for res in retrieval_results]
            llm_scores = await asyncio.gather(*tasks)
            
            # Update scores and sort
            for i, score in enumerate(llm_scores):
                final_results[i]["rerank_score"] = score
            
            final_results.sort(key=lambda x: x["rerank_score"], reverse=True)

    rerank_time = (time.time() - rerank_start) * 1000
    total_time = (time.time() - start_time) * 1000

    return {
        "results": final_results,
        "reranked": request.rerank,
        "metrics": {
            "retrieval_ms": round(retrieval_time, 2),
            "rerank_ms": round(rerank_time, 2),
            "total_ms": round(total_time, 2)
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
