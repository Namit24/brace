import json
import os
from typing import List, Dict, Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.utils import get_embedding, get_pinecone_index
from src.config import Config
from src.models import Actor
from dotenv import load_dotenv
load_dotenv()



class SearchEngine:
    def __init__(self) -> None:
        self.index = get_pinecone_index()
        self.actors_cache = self._load_actors_cache()
        self.allowed_ids = {a.unique_id for a in self.actors_cache}
        self.namespace = Config.NAMESPACE
        print(f"Loaded {len(self.actors_cache)} actors")

    def _load_actors_cache(self) -> List[Actor]:
        if not os.path.exists(Config.DATA_PATH):
            return []
        try:
            with open(Config.DATA_PATH, "r", encoding="utf-8") as f:
                return [Actor.model_validate(x) for x in json.load(f)]
        except:
            return []

    def _build_searchable_text(self, actor: Actor) -> str:
        parts = [
            actor.profile.name,
            actor.profile.headline or "",
            actor.profile.bio or "",
            actor.profile.location or "",
        ]
        for job in actor.professional.workexperience:
            parts.extend([job.title, job.companyname, job.description or ""])
        for edu in actor.professional.education:
            parts.extend([edu.school, edu.fieldofstudy or ""])
        return " ".join(parts).lower()

    def _keyword_filter(self, query: str) -> List[str]:
        q = query.lower()
        keywords = [
            "google","tesla","apple","microsoft","netflix","amazon","flipkart","swiggy","meta",
            "bangalore","blr","san francisco","sf","new york","nyc","london",
            "founder","ceo","co-founder","cto","vp","director",
            "ml","machine learning","ai","computer vision","nlp",
        ]
        active = [k for k in keywords if k in q]
        if not active:
            return []
        candidates = []
        for actor in self.actors_cache:
            txt = self._build_searchable_text(actor)
            if any(k in txt for k in active):
                candidates.append(actor.unique_id)
        return candidates

    def _semantic_only(self, query_vec, top_k):
        results = self.index.query(
            vector=query_vec,
            top_k=top_k,
            include_metadata=True,
            namespace=self.namespace,
        )
        final = []
        for m in results.matches:
            if m.id in self.allowed_ids:
                final.append(
                    {
                        "actor_id": m.id,
                        "score": round(m.score, 4),
                        "name": m.metadata.get("name", ""),
                        "headline": m.metadata.get("headline", ""),
                    }
                )
        return final

    def hybrid_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        vec = get_embedding(query)
        if not vec:
            return []
        candidate_ids = self._keyword_filter(query)
        if candidate_ids:
            try:
                resp = self.index.fetch(ids=candidate_ids, namespace=self.namespace)
                ids, vectors, metas = [], [], []
                for vid, v in resp.vectors.items():
                    if vid in self.allowed_ids:
                        ids.append(vid)
                        vectors.append(v.values)
                        metas.append(v.metadata)
                if vectors:
                    scores = cosine_similarity([vec], vectors)[0]
                    order = np.argsort(scores)[::-1][:top_k]
                    out = []
                    for idx in order:
                        out.append(
                            {
                                "actor_id": ids[idx],
                                "score": round(float(scores[idx]), 4),
                                "name": metas[idx].get("name", ""),
                                "headline": metas[idx].get("headline", ""),
                            }
                        )
                    if out:
                        return out
            except:
                pass
        return self._semantic_only(vec, top_k)

    def search_single_query(self, query: str) -> None:
        print(f"\nSearching for: '{query}'")
        results = self.hybrid_search(query)
        print("\n--- Results ---")
        for i, r in enumerate(results):
            print(f"{i+1}. {r['name']} (Score: {r['score']})")
            print(f"   {r['headline'][:100]}...")
            print("-" * 40)
        out = {"query": query, "results": results, "search_type": "hybrid"}
        Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = Config.OUTPUT_DIR / "cli_result.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved to {path}")

    def process_csv(self) -> None:
        if not Config.QUERIES_PATH.exists():
            print(f"Queries CSV not found at {Config.QUERIES_PATH}")
            return
        with open(Config.QUERIES_PATH, "r", encoding="utf-8") as f:
            lines = [x.strip() for x in f.readlines() if x.strip()]
        if lines and lines[0].lower() == "queries":
            lines = lines[1:]
        Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Processing {len(lines)} queries")
        for i, q in enumerate(lines):
            print(f"\n--- Query {i+1}: '{q}' ---")
            res = self.hybrid_search(q)
            out = {"query": q, "results": res, "search_type": "hybrid"}
            path = Config.OUTPUT_DIR / f"query_{i+1}_results.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
            print(f"Saved: {path.name}")
