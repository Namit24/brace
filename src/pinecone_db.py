"""
Pinecone vector database operations.
Implements multi-namespace strategy for semantic separation.
"""
import os
from typing import Dict, List, Any, Optional, Set
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "bracee-people-search"
DIMENSION = 3072  # Gemini embedding dimension

# Namespaces for different chunk types
NAMESPACE_EDUCATION = "education"
NAMESPACE_SKILLS = "skills"
NAMESPACE_COMPANIES = "companies"
NAMESPACE_LOCATION = "location"


class PineconeDB:
    """Pinecone database wrapper with multi-namespace support."""
    
    def __init__(self):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = None
    
    def create_index(self):
        """Create the Pinecone index if it doesn't exist."""
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        
        if INDEX_NAME not in existing_indexes:
            print(f"Creating index: {INDEX_NAME}")
            self.pc.create_index(
                name=INDEX_NAME,
                dimension=DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print(f"Index {INDEX_NAME} created successfully")
        else:
            print(f"Index {INDEX_NAME} already exists")
        
        self.index = self.pc.Index(INDEX_NAME)
        return self.index
    
    def get_index(self):
        """Get or create the index."""
        if self.index is None:
            self.index = self.pc.Index(INDEX_NAME)
        return self.index
    
    def upsert_vectors(
        self,
        vectors: List[Dict[str, Any]],
        namespace: str
    ):
        """
        Upsert vectors to a specific namespace.
        
        vectors should be list of dicts with:
            - id: unique vector ID
            - values: embedding vector
            - metadata: dict of metadata
        """
        index = self.get_index()
        
        # Batch upsert
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch, namespace=namespace)
        
        print(f"Upserted {len(vectors)} vectors to namespace: {namespace}")
    
    def query(
        self,
        vector: List[float],
        namespace: str,
        top_k: int = 50,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Query a specific namespace."""
        index = self.get_index()
        
        results = index.query(
            vector=vector,
            namespace=namespace,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )
        
        return [
            {
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata
            }
            for match in results.matches
        ]
    
    def query_multiple_namespaces(
        self,
        vector: List[float],
        namespaces: List[str],
        top_k: int = 50
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Query multiple namespaces and return results separately."""
        results = {}
        for ns in namespaces:
            results[ns] = self.query(vector, ns, top_k)
        return results
    
    def delete_namespace(self, namespace: str):
        """Delete all vectors in a namespace."""
        index = self.get_index()
        index.delete(delete_all=True, namespace=namespace)
        print(f"Deleted namespace: {namespace}")
    
    def get_stats(self) -> Dict:
        """Get index statistics."""
        index = self.get_index()
        return index.describe_index_stats()


def intersect_results(
    results_list: List[List[Dict[str, Any]]],
    key: str = "actor_id"
) -> Set[str]:
    """
    Get intersection of actor IDs across multiple result sets.
    Used for AND logic.
    """
    if not results_list:
        return set()
    
    sets = []
    for results in results_list:
        actor_ids = set()
        for r in results:
            aid = r.get("metadata", {}).get(key, r.get(key))
            if aid:
                actor_ids.add(aid)
        sets.append(actor_ids)
    
    # Intersection of all sets
    if sets:
        return sets[0].intersection(*sets[1:]) if len(sets) > 1 else sets[0]
    return set()


def union_results(
    results_list: List[List[Dict[str, Any]]],
    key: str = "actor_id"
) -> Set[str]:
    """
    Get union of actor IDs across multiple result sets.
    Used for OR logic.
    """
    all_ids = set()
    for results in results_list:
        for r in results:
            aid = r.get("metadata", {}).get(key, r.get(key))
            if aid:
                all_ids.add(aid)
    return all_ids


def aggregate_scores(
    results_by_namespace: Dict[str, List[Dict[str, Any]]],
    valid_ids: Set[str]
) -> Dict[str, float]:
    """
    Aggregate scores for valid actor IDs across namespaces.
    Returns dict of actor_id -> combined score.
    """
    scores = {}
    counts = {}
    
    for namespace, results in results_by_namespace.items():
        for r in results:
            aid = r.get("metadata", {}).get("actor_id", "")
            if aid in valid_ids:
                if aid not in scores:
                    scores[aid] = 0
                    counts[aid] = 0
                scores[aid] += r.get("score", 0)
                counts[aid] += 1
    
    # Average scores
    for aid in scores:
        if counts[aid] > 0:
            scores[aid] /= counts[aid]
    
    return scores
