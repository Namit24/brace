"""Run queries from queries.csv and output results."""
import asyncio
import json
import csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_processing import load_json
from src.retriever import PeopleRetriever

async def run_queries(
    queries_path: str,
    actors_path: str,
    output_path: str,
    evaluate: bool = True,
    debug: bool = False
):
    print("Loading data...")
    actors = load_json(actors_path)
    
    cache_path = Path(__file__).parent.parent / "data" / "profiles_cache.json"
    if cache_path.exists():
        profiles_cache = load_json(str(cache_path))
    else:
        from src.data_processing import ActorProcessor
        processor = ActorProcessor()
        profiles_cache = {}
        for actor in actors:
            profile = processor.get_full_profile(actor)
            profiles_cache[profile["actor_id"]] = profile
    
    queries = []
    with open(queries_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                queries.append(row[0].strip())
    
    print(f"Processing {len(queries)} queries...")
    
    retriever = PeopleRetriever(actors, profiles_cache)
    
    all_results = []
    evaluations = []
    
    for i, query in enumerate(queries):
        print(f"\n[{i+1}/{len(queries)}] Query: {query}")
        
        try:
            result = await retriever.search(query, top_k=10, use_reranking=True, debug=debug)
            
            output = {
                "query": query,
                "results": result["results"]
            }
            all_results.append(output)
            
            print(f"  Top results:")
            for j, r in enumerate(result.get("results_with_details", [])[:5]):
                print(f"    {j+1}. {r['name']} - {r.get('headline', '')[:60]}... (score: {r.get('score', 0):.2f})")
            
            if evaluate:
                eval_result = await retriever.evaluate_search(
                    query,
                    result.get("results_with_details", []),
                    result.get("parsed_intent", {})
                )
                evaluations.append({
                    "query": query,
                    "evaluation": eval_result
                })
                print(f"  Evaluation score: {eval_result.get('overall_score', 'N/A')}/10")
                if eval_result.get("issues"):
                    print(f"  Issues: {eval_result['issues'][:2]}")
        
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results.append({
                "query": query,
                "results": [],
                "error": str(e)
            })
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_file}")
    
    if evaluate and evaluations:
        eval_file = output_file.parent / "evaluations.json"
        with open(eval_file, 'w') as f:
            json.dump(evaluations, f, indent=2)
        print(f"Evaluations saved to {eval_file}")
        
        scores = [e["evaluation"].get("overall_score", 0) for e in evaluations if isinstance(e["evaluation"].get("overall_score"), (int, float))]
        if scores:
            avg_score = sum(scores) / len(scores)
            print(f"\nAverage evaluation score: {avg_score:.1f}/10")
    
    return all_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="data/queries.csv", help="Path to queries CSV")
    parser.add_argument("--actors", default="data/random_actors.json", help="Path to actors JSON")
    parser.add_argument("--output", default="output/results.json", help="Output path")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()
    
    asyncio.run(run_queries(
        args.queries,
        args.actors,
        args.output,
        evaluate=not args.no_eval,
        debug=args.debug
    ))