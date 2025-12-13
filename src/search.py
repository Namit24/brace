import json
import os
from typing import List, Dict

from src.utils import get_embedding
from src.config import Config
from src.models import Actor
from dotenv import load_dotenv
load_dotenv()


class SearchEngine:
    def __init__(self) -> None:
        self.actors = self._load_actors()
        print(f"Loaded {len(self.actors)} actors")

    # -------------------------
    # Data loading
    # -------------------------
    def _load_actors(self) -> List[Actor]:
        if not os.path.exists(Config.DATA_PATH):
            return []
        with open(Config.DATA_PATH, "r", encoding="utf-8") as f:
            return [Actor.model_validate(x) for x in json.load(f)]

    # -------------------------
    # Query understanding
    # -------------------------
    def extract_constraints(self, query: str) -> Dict[str, str]:
        q = query.lower()
        constraints = {}

        if "stanford" in q:
            constraints["education_school"] = "stanford"

        if "google" in q and "worked" in q:
            constraints["company"] = "google"

        if "san francisco" in q or "sf" in q:
            constraints["location"] = "san francisco"

        if "blr" in q or "bangalore" in q:
            constraints["location"] = "bengaluru"

        return constraints

    # -------------------------
    # Hard filtering
    # -------------------------
    def matches_constraints(self, actor: Actor, constraints: Dict[str, str]) -> bool:
        # Education filter
        if "education_school" in constraints:
            found = False
            for edu in actor.professional.education:
                if constraints["education_school"] in edu.school.lower():
                    found = True
                    break
            if not found:
                return False

        # Company filter
        if "company" in constraints:
            found = False
            for job in actor.professional.workexperience:
                if constraints["company"] in job.companyname.lower():
                    found = True
                    break
            if not found:
                return False

        # Location filter
        if "location" in constraints:
            if constraints["location"] not in (actor.profile.location or "").lower():
                return False

        return True

    # -------------------------
    # Text used for ranking
    # -------------------------
    def build_search_text(self, actor: Actor) -> str:
        parts = [
            actor.profile.name,
            actor.profile.headline or "",
            actor.profile.bio or "",
            actor.profile.location or "",
        ]

        for job in actor.professional.workexperience:
            parts.append(f"{job.title} at {job.companyname}")

        for edu in actor.professional.education:
            parts.append(f"Studied {edu.degree or ''} at {edu.school}")

        return " ".join(parts)

    # -------------------------
    # Search entry point
    # -------------------------
    def search_single_query(self, query: str, top_k: int = 5):
        print(f"\nSearching for: '{query}'\n")

        constraints = self.extract_constraints(query)

        # Step 1: Hard filter
        candidates = [
            actor for actor in self.actors
            if self.matches_constraints(actor, constraints)
        ]

        if not candidates:
            print("No results after applying constraints.\n")
            return []

        # Step 2: Embed query
        query_vec = get_embedding(query)
        if not query_vec:
            print("Failed to embed query.")
            return []

        # Step 3: Rank candidates
        scored = []
        for actor in candidates:
            text = self.build_search_text(actor)
            vec = get_embedding(text)
            if not vec:
                continue

            score = self.cosine_similarity(query_vec, vec)
            scored.append((actor, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        results = scored[:top_k]

        # Output
        for i, (actor, score) in enumerate(results, 1):
            print(f"{i}. {actor.profile.name} (Score: {score:.4f})")
            print(f"   {actor.profile.headline}")
            print("-" * 40)

        return results

    # -------------------------
    # Utils
    # -------------------------
    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        return dot / (norm_a * norm_b + 1e-8)
