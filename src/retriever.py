"""Main retrieval system for people search."""
import asyncio
from typing import Dict, List, Any, Set, Optional
from .data_processing import load_json
from .embeddings import get_query_embedding, get_embeddings
from .pinecone_db import (
    PineconeDB, 
    NAMESPACE_EDUCATION, 
    NAMESPACE_SKILLS, 
    NAMESPACE_COMPANIES, 
    NAMESPACE_LOCATION,
    intersect_results,
    union_results,
    aggregate_scores
)
from .llm import normalize_and_parse_query, rerank_results, evaluate_results


class PeopleRetriever:
    """Main retrieval system for people search."""
    
    def __init__(self, actors_data: List[Dict], profiles_cache: Dict[str, Dict]):
        self.db = PineconeDB()
        self.actors_data = actors_data
        self.profiles_cache = profiles_cache
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        use_reranking: bool = True,
        debug: bool = False
    ) -> Dict[str, Any]:
        """
        Main search method.
        
        1. Normalize and parse query using Gemini (dynamic expansion)
        2. Query appropriate namespaces with expanded terms
        3. Apply AND/OR logic
        4. Rerank with LLM judge
        5. Return ranked results
        """
        # Step 1: Dynamic query normalization with Gemini
        parsed = await normalize_and_parse_query(query)
        
        if debug:
            print(f"Parsed intent: {parsed}")
        
        # Step 2: Build query vectors for each filter type
        results_by_category = {}
        
        # Education filter - use all expanded variations with canonical groups
        if parsed.get("education"):
            edu_results = await self._search_education(
                parsed["education"],
                parsed.get("education_logic", "OR"),
                parsed.get("education_groups", [])
            )
            if edu_results:
                results_by_category["education"] = edu_results
        
        # Skills filter - already semantically expanded by Gemini
        if parsed.get("skills"):
            skill_results = await self._search_skills(
                parsed["skills"],
                parsed.get("skills_logic", "OR"),
                parsed.get("normalized_query", query)
            )
            if skill_results:
                results_by_category["skills"] = skill_results
        
        # Companies filter
        if parsed.get("companies"):
            company_results = await self._search_companies(
                parsed["companies"],
                parsed.get("companies_logic", "OR")
            )
            if company_results:
                results_by_category["companies"] = company_results
        
        # Location filter - use expanded variations
        if parsed.get("locations"):
            location_results = await self._search_locations(
                parsed["locations"],
                parsed.get("locations_logic", "OR")
            )
            if location_results:
                results_by_category["location"] = location_results
        
        # Step 3: Combine results across categories (always AND)
        if not results_by_category:
            # Fallback: use normalized query for general search
            normalized = parsed.get("normalized_query", query)
            embedding = await get_query_embedding(normalized)
            raw_results = self.db.query(embedding, NAMESPACE_SKILLS, top_k=50)
            valid_ids = {r["metadata"]["actor_id"] for r in raw_results if r.get("metadata")}
            scores = {r["metadata"]["actor_id"]: r["score"] for r in raw_results if r.get("metadata")}
        else:
            # Intersect all category results (AND logic across categories)
            all_result_sets = [
                [(r["actor_id"], r["score"]) for r in results]
                for results in results_by_category.values()
            ]
            
            if len(all_result_sets) == 1:
                valid_ids = {r[0] for r in all_result_sets[0]}
                scores = {r[0]: r[1] for r in all_result_sets[0]}
            else:
                # Intersection
                id_sets = [set(r[0] for r in rs) for rs in all_result_sets]
                valid_ids = id_sets[0].intersection(*id_sets[1:])
                
                # Aggregate scores for valid IDs
                scores = {}
                for result_set in all_result_sets:
                    for actor_id, score in result_set:
                        if actor_id in valid_ids:
                            scores[actor_id] = scores.get(actor_id, 0) + score
                
                # Average
                for aid in scores:
                    scores[aid] /= len(all_result_sets)
        
        if debug:
            print(f"Valid IDs after filtering: {len(valid_ids)}")
        
        # Step 4: Build candidate list with profiles
        candidates = []
        for actor_id in valid_ids:
            if actor_id in self.profiles_cache:
                profile = self.profiles_cache[actor_id].copy()
                profile["score"] = scores.get(actor_id, 0.5)
                candidates.append(profile)
        
        # Sort by initial score
        candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
        candidates = candidates[:top_k * 2]  # Take more for reranking
        
        # Step 5: Rerank with LLM judge
        if use_reranking and candidates:
            candidates = await rerank_results(query, candidates, parsed)
        
        # Final sort and trim
        candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
        final_results = candidates[:top_k]
        
        return {
            "query": query,
            "parsed_intent": parsed,
            "results": [
                {"actor_id": r["actor_id"], "score": round(r.get("score", 0), 3)}
                for r in final_results
            ],
            "results_with_details": final_results
        }
    
    async def _search_education(
        self,
        schools: List[str],
        logic: str,
        education_groups: List[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Search education namespace with pre-expanded school names."""
        query_text = f"Studied at {' '.join(schools)}"
        embedding = await get_query_embedding(query_text)
        results = self.db.query(embedding, NAMESPACE_EDUCATION, top_k=100)
        
        schools_lower = [s.lower() for s in schools]
        
        filtered = []
        seen_actors = set()
        
        for r in results:
            school_in_result = r.get("metadata", {}).get("school", "").lower()
            actor_id = r["metadata"]["actor_id"]
            
            if any(s in school_in_result or school_in_result in s for s in schools_lower):
                if actor_id not in seen_actors:
                    seen_actors.add(actor_id)
                    filtered.append({
                        "actor_id": actor_id,
                        "score": r["score"],
                        "school": r["metadata"].get("school", "")
                    })
        
        # For AND logic, use canonical groups from LLM (robust for multi-word schools)
        if logic == "AND" and education_groups and len(education_groups) > 1:
            from collections import defaultdict
            
            # Build actor -> schools mapping from raw results
            actor_schools = defaultdict(set)
            for r in results:
                actor_id = r["metadata"]["actor_id"]
                school = r["metadata"].get("school", "").lower()
                actor_schools[actor_id].add(school)
            
            # Check each actor has at least one school from EACH canonical group
            valid_actors = set()
            for actor_id, actor_school_set in actor_schools.items():
                matched_groups = 0
                for group in education_groups:
                    variations = [v.lower() for v in group.get("variations", [])]
                    # Check if actor has any school matching this group
                    if any(
                        any(var in asch or asch in var for var in variations)
                        for asch in actor_school_set
                    ):
                        matched_groups += 1
                
                if matched_groups == len(education_groups):
                    valid_actors.add(actor_id)
            
            filtered = [f for f in filtered if f["actor_id"] in valid_actors]
        
        return filtered
    
    async def _search_skills(
        self,
        skills: List[str],
        logic: str,
        normalized_query: str = ""
    ) -> List[Dict[str, Any]]:
        """Search skills namespace with pre-expanded skills."""
        if normalized_query:
            query_text = f"Skills: {normalized_query}"
        else:
            query_text = f"Skills and expertise in: {', '.join(skills)}"
        
        embedding = await get_query_embedding(query_text)
        results = self.db.query(embedding, NAMESPACE_SKILLS, top_k=100)
        
        return [
            {
                "actor_id": r["metadata"]["actor_id"],
                "score": r["score"],
                "detected_skills": r["metadata"].get("detected_skills", [])
            }
            for r in results
            if r.get("metadata")
        ]
    
    async def _search_companies(
        self,
        companies: List[str],
        logic: str
    ) -> List[Dict[str, Any]]:
        """Search companies namespace with pre-expanded company names."""
        query_text = f"Worked at {' '.join(companies)}"
        embedding = await get_query_embedding(query_text)
        results = self.db.query(embedding, NAMESPACE_COMPANIES, top_k=100)
        
        companies_lower = [c.lower() for c in companies]
        
        filtered = []
        seen_actors = set()
        
        for r in results:
            companies_in_result = r.get("metadata", {}).get("companies", [])
            companies_str = ' '.join(companies_in_result).lower()
            actor_id = r["metadata"]["actor_id"]
            
            if any(c in companies_str for c in companies_lower):
                if actor_id not in seen_actors:
                    seen_actors.add(actor_id)
                    filtered.append({
                        "actor_id": actor_id,
                        "score": r["score"],
                        "companies": companies_in_result
                    })
        
        return filtered
    
    async def _search_locations(
        self,
        locations: List[str],
        logic: str
    ) -> List[Dict[str, Any]]:
        """Search location namespace with pre-expanded location names."""
        query_text = f"Located in {' '.join(locations)}"
        embedding = await get_query_embedding(query_text)
        results = self.db.query(embedding, NAMESPACE_LOCATION, top_k=100)
        
        locations_lower = [loc.lower() for loc in locations]
        
        filtered = []
        seen_actors = set()
        
        for r in results:
            loc_in_result = r.get("metadata", {}).get("location", "").lower()
            actor_id = r["metadata"]["actor_id"]
            
            if any(loc in loc_in_result or loc_in_result in loc for loc in locations_lower):
                if actor_id not in seen_actors:
                    seen_actors.add(actor_id)
                    filtered.append({
                        "actor_id": actor_id,
                        "score": r["score"],
                        "location": r["metadata"].get("location", "")
                    })
        
        return filtered
    
    async def evaluate_search(
        self,
        query: str,
        results: List[Dict[str, Any]],
        parsed_intent: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate search quality using LLM judge."""
        return await evaluate_results(query, results, parsed_intent)
