import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    EMBEDDING_MODEL = "google/gemini-embedding-001"

    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    INDEX_NAME = "brace"
    NAMESPACE = "actors"
    DIMENSION = 3072    
    CLOUD = "aws"
    REGION = "us-east-1"

    DATA_PATH = Path("data") / "random_actors.json"
    QUERIES_PATH = Path("data") / "queries.csv"
    OUTPUT_DIR = Path("output")

    if not OPENROUTER_API_KEY:
        raise ValueError("Missing OPENROUTER_API_KEY")
    if not PINECONE_API_KEY:
        raise ValueError("Missing PINECONE_API_KEY")
