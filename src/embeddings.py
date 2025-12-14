import httpx
import os
from typing import List
from dotenv import load_dotenv
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
EMBEDDING_MODEL = "google/gemini-embedding-001"
EMBEDDING_URL = "https://openrouter.ai/api/v1/embeddings"
async def get_embeddings(texts: List[str], task_type: str = "retrieval_document") -> List[List[float]]:
    """
    Get embeddings for a list of texts using Gemini embedding model via OpenRouter.
    
    Args:
        texts: List of strings to embed
        task_type: One of 'retrieval_document', 'retrieval_query', 'semantic_similarity'
    
    Returns:
        List of embedding vectors
    """
    if not texts:
        return []
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://bracee.local",
        "X-Title": "Bracee Semantic Search"
    }
    
    embeddings = []
    batch_size = 20
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            payload = {
                "model": EMBEDDING_MODEL,
                "input": batch,
            }
            
            response = await client.post(EMBEDDING_URL, headers=headers, json=payload)
            response.raise_for_status()
            
            data = response.json()
            batch_embeddings = [item["embedding"] for item in data["data"]]
            embeddings.extend(batch_embeddings)
    
    return embeddings


async def get_query_embedding(query: str) -> List[float]:
    embeddings = await get_embeddings([query], task_type="retrieval_query")
    return embeddings[0] if embeddings else []


def get_embeddings_sync(texts: List[str]) -> List[List[float]]:
    import asyncio
    return asyncio.run(get_embeddings(texts))


def get_query_embedding_sync(query: str) -> List[float]:
    import asyncio
    return asyncio.run(get_query_embedding(query))
