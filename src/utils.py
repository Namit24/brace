import os
import json
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX", "brace")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)
def get_embedding(text: str):
    text = text.strip().replace("\n", " ")

    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding

    except Exception as e:
        print("[ERROR] Failed to generate embedding:", e)
        raise e

def get_pinecone_index():
    """
    Returns a Pinecone index object.
    Automatically performs a safe delete-all without namespace errors.
    """
    pc = Pinecone(api_key=PINECONE_API_KEY)

    try:
        index = pc.Index(INDEX_NAME)
    except Exception as e:
        print(f"[ERROR] Could not connect to Pinecone index '{INDEX_NAME}':", e)
        raise e

    try:
        index.delete(delete_all=True)
        print("[INFO] Index cleared.")
    except Exception as e:
        print("[WARN] Could not clear index (probably empty):", e)

    return index
