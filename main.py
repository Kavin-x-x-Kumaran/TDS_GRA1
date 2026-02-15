import os
import time
import asyncio
import numpy as np
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional

# --- Configuration ---
AI_PIPE_TOKEN = os.getenv("AI_PIPE_TOKEN")
BASE_URL = "https://aipipe.org/openrouter/v1"
CHAT_URL = f"{BASE_URL}/chat/completions"
EMBED_URL = f"{BASE_URL}/embeddings"

app = FastAPI(title="SearchTech Solutions Pipeline")

# 1. FIX: Enable CORS so the AI Checker can reach you
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data & Embeddings ---
DOCS = [{"id": i, "text": f"Abstract {i}: Scientific discovery regarding AI in search engines focusing on {topic}."}
        for i, topic in enumerate(["vector databases", "re-ranking", "transformers", "LLMs", "latency"] * 25)]
DOC_TEXTS = [d["text"] for d in DOCS]
doc_embeddings = None

async def get_embeddings(texts: List[str]):
    async with httpx.AsyncClient() as client:
        res = await client.post(EMBED_URL, headers={"Authorization": f"Bearer {AI_PIPE_TOKEN}"},
                                json={"model": "text-embedding-3-small", "input": texts}, timeout=30.0)
        return np.array([item["embedding"] for item in res.json()["data"]])

@app.on_event("startup")
async def startup():
    global doc_embeddings
    # Pre-compute to meet the <200ms requirement
    doc_embeddings = await get_embeddings(DOC_TEXTS)
    doc_embeddings /= np.linalg.norm(doc_embeddings, axis=1, keepdims=True)

# 2. FIX: Add Root Route
@app.get("/")
async def root():
    return {"status": "running", "endpoint": "/search"}

# 3. FIX: Match the exact schema required by SearchTech
class SearchRequest(BaseModel):
    query: str
    k: int = 8
    rerank: bool = True
    rerankK: int = 5

@app.post("/search")
async def search(req: SearchRequest):
    start_total = time.time()
    
    # Stage 1: Vector Search
    query_vec = await get_embeddings([req.query])
    query_vec /= np.linalg.norm(query_vec)
    scores = np.dot(doc_embeddings, query_vec.T).flatten()
    top_idx = np.argsort(scores)[::-1][:req.k]
    
    results = [{"id": DOCS[i]["id"], "score": float(scores[i]), "content": DOCS[i]["text"]} for i in top_idx]
    latency_retrieval = (time.time() - start_total) * 1000

    # Stage 2: Re-ranking
    if req.rerank:
        async with httpx.AsyncClient() as client:
            tasks = []
            for r in results:
                prompt = f"Query: {req.query}\nDocument: {r['content']}\n\nRate the relevance of this document to the query on a scale of 0-10. Respond with only the number."
                tasks.append(client.post(CHAT_URL, headers={"Authorization": f"Bearer {AI_PIPE_TOKEN}"},
                                         json={"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": prompt}], "temperature": 0}))
            responses = await asyncio.gather(*tasks)
            for i, res in enumerate(responses):
                val = res.json()['choices'][0]['message']['content'].strip()
                results[i]["score"] = float(val) / 10.0 # Normalize 0-1
        
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:req.rerankK] # Return top 5 as requested

    return {
        "results": results,
        "reranked": req.rerank,
        "metrics": {
            "latency": round((time.time() - start_total) * 1000, 2),
            "totalDocs": 125
        }
    }
