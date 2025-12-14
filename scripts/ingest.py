import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_processing import ActorProcessor, load_json
from src.embeddings import get_embeddings
from src.pinecone_db import (
    PineconeDB,
    NAMESPACE_EDUCATION,
    NAMESPACE_SKILLS,
    NAMESPACE_COMPANIES,
    NAMESPACE_LOCATION
)

async def ingest_actors(actors_path: str, reset: bool = False):
    print("Loading data...")
    actors = load_json(actors_path)
    processor = ActorProcessor()
    
    print(f"Processing {len(actors)} actors...")
    processed = processor.process_all_actors(actors)
    
    db = PineconeDB()
    db.create_index()
    
    if reset:
        print("Resetting namespaces...")
        for ns in [NAMESPACE_EDUCATION, NAMESPACE_SKILLS, NAMESPACE_COMPANIES, NAMESPACE_LOCATION]:
            try:
                db.delete_namespace(ns)
            except Exception as e:
                print(f"Could not delete {ns}: {e}")
    
    education_chunks = []
    skills_chunks = []
    companies_chunks = []
    location_chunks = []
    
    for p in processed:
        education_chunks.extend(p["education_chunks"])
        if p["skills_chunk"]:
            skills_chunks.append(p["skills_chunk"])
        if p["companies_chunk"]:
            companies_chunks.append(p["companies_chunk"])
        if p["location_chunk"]:
            location_chunks.append(p["location_chunk"])
    
    print(f"Chunks to embed:")
    print(f"  - Education: {len(education_chunks)}")
    print(f"  - Skills: {len(skills_chunks)}")
    print(f"  - Companies: {len(companies_chunks)}")
    print(f"  - Location: {len(location_chunks)}")
    
    await ingest_namespace(db, education_chunks, NAMESPACE_EDUCATION)
    await ingest_namespace(db, skills_chunks, NAMESPACE_SKILLS)
    await ingest_namespace(db, companies_chunks, NAMESPACE_COMPANIES)
    await ingest_namespace(db, location_chunks, NAMESPACE_LOCATION)
    
    profiles_cache = {p["actor_id"]: p["profile"] for p in processed}
    cache_path = Path(__file__).parent.parent / "data" / "profiles_cache.json"
    with open(cache_path, 'w') as f:
        json.dump(profiles_cache, f, indent=2)
    print(f"Saved profiles cache to {cache_path}")
    
    print("\nIngestion complete!")
    print(db.get_stats())


async def ingest_namespace(
    db: PineconeDB,
    chunks: List[Dict[str, Any]],
    namespace: str
):
    if not chunks:
        print(f"No chunks for {namespace}")
        return
    
    print(f"Embedding {len(chunks)} chunks for {namespace}...")
    
    texts = [c["text"] for c in chunks]
    embeddings = await get_embeddings(texts)
    
    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vector_id = f"{namespace}_{chunk['actor_id']}_{i}"
        
        metadata = {k: v for k, v in chunk.items() if k != "text"}
        
        vectors.append({
            "id": vector_id,
            "values": embedding,
            "metadata": metadata
        })
    
    db.upsert_vectors(vectors, namespace)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--actors", default="data/random_actors.json", help="Path to actors JSON")
    parser.add_argument("--reset", action="store_true", help="Reset namespaces before ingestion")
    args = parser.parse_args()
    
    asyncio.run(ingest_actors(args.actors, args.reset))