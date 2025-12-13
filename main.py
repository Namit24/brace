import sys
from dotenv import load_dotenv
load_dotenv()

from src.ingestion import load_actors, vectorize_actors, upload_vectors
from src.search import SearchEngine


def run_interactive_cli(engine: SearchEngine):
    print("Interactive Search Mode (Pinecone Only)")
    print("Type a query or 'exit' to quit.\n")

    while True:
        query = input("Query: ").strip()

        if query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if not query:
            print("Please type something.\n")
            continue

        print("\nSearching...\n")
        engine.search_single_query(query)


def main():
    args = sys.argv[1:]

    skip_ingest = "--skip-ingest" in args
    interactive = "--interactive" in args

    query_args = [arg for arg in args if not arg.startswith("--")]
    custom_query = " ".join(query_args) if query_args else None

    if not skip_ingest:
        actors = load_actors()
        vectors = vectorize_actors(actors)
        upload_vectors(vectors)
    else:
        print("=== SKIPPING INGESTION ===\n")

    engine = SearchEngine()

    if interactive:
        run_interactive_cli(engine)
    elif custom_query:
        engine.search_single_query(custom_query)
    else:
        print("No query provided.")



if __name__ == "__main__":
    main()