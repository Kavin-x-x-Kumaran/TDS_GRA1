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
EMBED_MODEL = "all-MiniLM-L6-v2"

app = FastAPI(title="SearchTech Two-Stage Pipeline")

# --- In-Memory Store & Model Initialization ---
DOCS = [
    {"id": i, "text": f"Abstract {i}: Scientific discovery regarding AI in search engines focusing on {topic}."}
    for i, topic in enumerate(["vector databases", "re-ranking", "transformers", "LLMs", "latency"] * 25)
]
DOC_TEXTS = [d["text"] for d in DOCS]

# Global variables for model and embeddings
model = None
doc_embeddings = None

@app.on_event("startup")
async def load_model():
    global model, doc_embeddings
    print("==> Loading Transformer Model...")
    # Loading the model here ensures it only happens once when the server starts
    model = SentenceTransformer(EMBED_MODEL)
    print("==> Encoding 125 documents...")
    embeddings = model.encode(DOC_TEXTS, convert_to_numpy=True)
    # Normalize for Cosine Similarity
    doc_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    print("==> System Ready for Search Tech Solutions.")

# --- NEW: Health Check Endpoint ---
@app.get("/health")
async def health_check():
    """
    Endpoint for UptimeRobot or cron-job.org to ping 
    to prevent the Render free tier from sleeping.
    """
    return {
        "status": "online",
        "model_loaded": model is not None,
        "uptime_check": True,
        "timestamp": time.time()
    }

# --- Search Logic ---
class SearchRequest(BaseModel):
    query: str
    rerank: bool = True

class SearchResponse(BaseModel):
    results: List[Dict]
    reranked: bool
    metrics: Dict

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
        # Basic parsing of the LLM response
        return float(score_raw) / 10.0
    except Exception:
        return 0.0

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model still loading, please wait.")
        
    start_time = time.time()
    
    # STAGE 1: Vector Search (Retrieval)
    query_embedding = model.encode([request.query], convert_to_numpy=True)
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    
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
    
    # STAGE 2: Re-ranking (LLM)
    rerank_start = time.time()
    if request.rerank:
        async with httpx.AsyncClient() as client:
            tasks = [get_llm_score(client, request.query, res["text"]) for res in retrieval_results]
            llm_scores = await asyncio.gather(*tasks)
            for i, score in enumerate(llm_scores):
                retrieval_results[i]["rerank_score"] = score
            # Re-sort based on LLM reasoning
            retrieval_results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

    rerank_time = (time.time() - rerank_start) * 1000
    
    return {
        "results": retrieval_results,
        "reranked": request.rerank,
        "metrics": {
            "retrieval_ms": round(retrieval_time, 2),
            "rerank_ms": round(rerank_time, 2),
            "total_ms": round((time.time() - start_time) * 1000, 2)
        }
    }
