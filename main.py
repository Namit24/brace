#!/usr/bin/env python3
import asyncio
import argparse
import json
import sys
from pathlib import Path

from src.data_processing import ActorProcessor, load_json
from src.embeddings import get_embeddings
from src.pinecone_db import (
    PineconeDB,
    NAMESPACE_EDUCATION,
    NAMESPACE_SKILLS,
    NAMESPACE_COMPANIES,
    NAMESPACE_LOCATION
)
from src.retriever import PeopleRetriever


async def run_ingestion(actors_path: str, reset: bool = False) -> dict:
    print("\n" + "="*60)
    print("INGESTION PHASE")
    print("="*60)
    
    print(f"\nLoading data from {actors_path}...")
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
                print(f"  Could not delete {ns}: {e}")
    
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
    
    print(f"\nChunks to embed:")
    print(f"  Education: {len(education_chunks)}")
    print(f"  Skills:    {len(skills_chunks)}")
    print(f"  Companies: {len(companies_chunks)}")
    print(f"  Location:  {len(location_chunks)}")
    
    for chunks, namespace, name in [
        (education_chunks, NAMESPACE_EDUCATION, "Education"),
        (skills_chunks, NAMESPACE_SKILLS, "Skills"),
        (companies_chunks, NAMESPACE_COMPANIES, "Companies"),
        (location_chunks, NAMESPACE_LOCATION, "Location"),
    ]:
        if chunks:
            print(f"\nEmbedding {len(chunks)} {name} chunks...")
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
    
    profiles_cache = {p["actor_id"]: p["profile"] for p in processed}
    cache_path = Path(__file__).parent / "data" / "profiles_cache.json"
    with open(cache_path, 'w') as f:
        json.dump(profiles_cache, f, indent=2)
    
    print(f"\nSaved profiles cache to {cache_path}")
    print(f"\n{db.get_stats()}")
    print("\nIngestion complete!")
    
    return {"actors": actors, "profiles_cache": profiles_cache}


async def run_single_query(query: str, actors: list, profiles_cache: dict, debug: bool = False):
    retriever = PeopleRetriever(actors, profiles_cache)
    
    print(f"\nQuery: {query}")
    print("-" * 50)
    
    result = await retriever.search(query, top_k=10, use_reranking=True, debug=debug)
    
    print(f"\nTop Results:")
    for i, r in enumerate(result.get("results_with_details", [])[:10]):
        score = r.get("score", 0)
        name = r.get("name", "Unknown")
        headline = r.get("headline", "")[:60]
        location = r.get("location", "")
        education = ", ".join(r.get("education", [])[:2])
        
        print(f"\n  {i+1}. {name} (score: {score:.2f})")
        print(f"     {headline}...")
        if location:
            print(f"     {location}")
        if education:
            print(f"     {education}")
    
    return result


async def interactive_mode(actors: list, profiles_cache: dict):
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print("\nEnter queries to search. Type 'quit' or 'exit' to stop.")
    print("Type 'debug' to toggle debug mode.")
    print("-" * 60)
    
    debug = False
    
    while True:
        try:
            query = input("\nQuery: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ('quit', 'exit', 'q'):
                print("\nGoodbye!")
                break
            
            if query.lower() == 'debug':
                debug = not debug
                print(f"Debug mode: {'ON' if debug else 'OFF'}")
                continue
            
            await run_single_query(query, actors, profiles_cache, debug=debug)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


async def main():
    parser = argparse.ArgumentParser(
        description="Bracee People Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --interactive
  python main.py --interactive --skip-ingest
  python main.py --query "frontend devs"
  python main.py --reset
        """
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true"
    )
    parser.add_argument(
        "--skip-ingest", "-s",
        action="store_true"
    )
    parser.add_argument(
        "--query", "-q",
        type=str
    )
    parser.add_argument(
        "--actors",
        default="data/random_actors.json"
    )
    parser.add_argument(
        "--reset",
        action="store_true"
    )
    parser.add_argument(
        "--debug",
        action="store_true"
    )
    
    args = parser.parse_args()
    
    actors_path = Path(__file__).parent / args.actors
    cache_path = Path(__file__).parent / "data" / "profiles_cache.json"
    
    actors = None
    profiles_cache = None
    
    if not args.skip_ingest:
        data = await run_ingestion(str(actors_path), reset=args.reset)
        actors = data["actors"]
        profiles_cache = data["profiles_cache"]
    else:
        print("\nSkipping ingestion (using existing index)")
        
        if actors_path.exists():
            actors = load_json(str(actors_path))
        else:
            print(f"Error: Actors file not found: {actors_path}")
            sys.exit(1)
        
        if cache_path.exists():
            profiles_cache = load_json(str(cache_path))
        else:
            print("Profiles cache not found, generating...")
            processor = ActorProcessor()
            profiles_cache = {}
            for actor in actors:
                profile = processor.get_full_profile(actor)
                profiles_cache[profile["actor_id"]] = profile
    
    if args.query:
        await run_single_query(args.query, actors, profiles_cache, debug=args.debug)
        return
    
    if args.interactive:
        await interactive_mode(actors, profiles_cache)
    elif args.skip_ingest:
        print("\nTip: Use --interactive or --query to search")


if __name__ == "__main__":
    asyncio.run(main())
