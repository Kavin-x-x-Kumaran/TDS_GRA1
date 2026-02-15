import os
import time
import asyncio
import numpy as np
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

# --- Configuration ---
AI_PIPE_TOKEN = os.getenv("AI_PIPE_TOKEN", "your_token_here")
# AI PIPE standard endpoints
BASE_URL = "https://aipipe.org/openrouter/v1"
CHAT_URL = f"{BASE_URL}/chat/completions"
EMBED_URL = f"{BASE_URL}/embeddings"

# Light but powerful models
EMBED_MODEL = "text-embedding-3-small" 
RERANK_MODEL = "openai/gpt-4o-mini"

app = FastAPI(title="SearchTech Lightweight Pipeline")

# --- In-Memory Store ---
DOCS = [
    {"id": i, "text": f"Abstract {i}: Scientific discovery regarding AI in search engines focusing on {topic}."}
    for i, topic in enumerate(["vector databases", "re-ranking", "transformers", "LLMs", "latency"] * 25)
]
DOC_TEXTS = [d["text"] for d in DOCS]

# Global cache for embeddings
doc_embeddings = None

# --- Helper: AI PIPE Wrapper ---
async def call_aipipe_embed(texts: List[str]):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            EMBED_URL,
            headers={"Authorization": f"Bearer {AI_PIPE_TOKEN}"},
            json={"model": EMBED_MODEL, "input": texts},
            timeout=30.0
        )
        if response.status_code != 200:
            raise Exception(f"AI PIPE Error: {response.text}")
        data = response.json()["data"]
        return np.array([item["embedding"] for item in data])

@app.on_event("startup")
async def startup_event():
    global doc_embeddings
    print("==> Requesting embeddings from AI PIPE for 125 docs...")
    # We do this once so searches are fast later
    doc_embeddings = await call_aipipe_embed(DOC_TEXTS)
    # Normalize for Cosine Similarity
    doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    print("==> System Ready.")

@app.get("/health")
def health():
    return {"status": "online", "memory_friendly": True}

# --- Search Logic ---
class SearchRequest(BaseModel):
    query: str
    rerank: bool = True

async def get_llm_score(client: httpx.AsyncClient, query: str, doc_text: str):
    prompt = f"Query: {query}\nDoc: {doc_text}\nRate relevance 0-10. Respond with ONLY the number."
    try:
        res = await client.post(
            CHAT_URL,
            headers={"Authorization": f"Bearer {AI_PIPE_TOKEN}"},
            json={"model": RERANK_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0},
            timeout=5.0
        )
        return float(res.json()['choices'][0]['message']['content'].strip()) / 10.0
    except:
        return 0.0

@app.post("/search")
async def search(request: SearchRequest):
    start_time = time.time()
    
    # STAGE 1: Vector Retrieval (via Cloud Embedding)
    query_vec = await call_aipipe_embed([request.query])
    query_vec = query_vec / np.linalg.norm(query_vec)
    
    scores = np.dot(doc_embeddings, query_vec.T).flatten()
    top_indices = np.argsort(scores)[::-1][:8]
    
    results = [{"id": DOCS[i]["id"], "text": DOCS[i]["text"], "score": float(scores[i])} for i in top_indices]
    retrieval_ms = (time.time() - start_time) * 1000
    
    # STAGE 2: Re-ranking
    rerank_start = time.time()
    if request.rerank:
        async with httpx.AsyncClient() as client:
            tasks = [get_llm_score(client, request.query, r["text"]) for r in results]
            llm_scores = await asyncio.gather(*tasks)
            for i, s in enumerate(llm_scores):
                results[i]["rerank_score"] = s
            results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            
    return {
        "results": results,
        "metrics": {
            "retrieval_ms": round(retrieval_ms, 2),
            "rerank_ms": round((time.time() - rerank_start) * 1000, 2),
            "total_ms": round((time.time() - start_time) * 1000, 2)
        }
    }
