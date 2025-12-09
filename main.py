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

    # Extract query arguments (words that are not flags)
    query_args = [arg for arg in args if not arg.startswith("--")]
    custom_query = " ".join(query_args) if query_args else None

    # ---------------------------
    # Step 1: Ingest / Skip ingest
    # ---------------------------
    if not skip_ingest:
        actors = load_actors()
        if actors:
            vectors = vectorize_actors(actors)
            upload_vectors(vectors)
        else:
            print("No actors found to ingest.\n")
    else:
        print("=== SKIPPING INGESTION ===\n")

    # ---------------------------
    # Step 2: Create the Search Engine
    # ---------------------------
    engine = SearchEngine()

    # ---------------------------
    # Step 3: Interactive CLI Mode
    # ---------------------------
    if interactive:
        return run_interactive_cli(engine)
    if custom_query:
        return engine.search_single_query(custom_query)
    engine.process_csv()


if __name__ == "__main__":
    main()