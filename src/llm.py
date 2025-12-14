import httpx
import os
import json
import hashlib
from functools import lru_cache
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
LLM_MODEL = "google/gemini-2.5-flash"
LLM_URL = "https://openrouter.ai/api/v1/chat/completions"

_query_cache: Dict[str, Dict[str, Any]] = {}
_CACHE_MAX_SIZE = 1000


def _get_cache_key(query: str) -> str:
    return hashlib.md5(query.lower().strip().encode()).hexdigest()


async def call_llm(messages: List[Dict[str, str]], response_format: Optional[str] = None) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://bracee.local",
        "X-Title": "Bracee Semantic Search"
    }
    
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 2000,
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(LLM_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


async def normalize_and_parse_query(query: str, use_cache: bool = True) -> Dict[str, Any]:
    """
    Dynamically normalize and parse a query using Gemini.
    Uses caching to reduce LLM calls for repeated queries.
    
    Returns dict with:
        - education: list of schools with canonical grouping
        - skills: list of skills/roles to match (semantically expanded)
        - companies: list of companies to match (with variations)
        - locations: list of locations to match (with variations)
        - *_logic: 'AND' or 'OR' for combining criteria
        - education_groups: canonical groupings for AND logic
        - normalized_query: cleaned up query for embedding
    """
    cache_key = _get_cache_key(query)
    if use_cache and cache_key in _query_cache:
        return _query_cache[cache_key]
    
    from .aliases import get_alias_context_for_prompt
    alias_context = get_alias_context_for_prompt()
    
    system_prompt = f"""You are a query normalizer for a LinkedIn people search system. Your job is to:
1. Extract structured filters from natural language
2. EXPAND all abbreviations, acronyms, and aliases to their full forms AND common variations
3. Apply correct AND/OR logic based on user intent
4. Provide CANONICAL groupings for schools (for AND logic verification)

{alias_context}

## CRITICAL RULES

### School Canonicalization (IMPORTANT for AND logic):
When multiple schools are mentioned with AND logic, group them by canonical ID:
- "Stanford and MIT" → education_groups: [
    {{"canonical": "stanford", "variations": ["Stanford", "Stanford University"]}},
    {{"canonical": "mit", "variations": ["MIT", "Massachusetts Institute of Technology"]}}
  ]
- "IIT Bombay" → education_groups: [{{"canonical": "iit_bombay", "variations": ["IIT Bombay", "Indian Institute of Technology Bombay", "IITB"]}}]
- "IISc" → education_groups: [{{"canonical": "iisc", "variations": ["IISc", "Indian Institute of Science", "IISc Bangalore"]}}]

### Abbreviation/Alias Expansion:
LOCATIONS: Expand to all variations (blr → Bangalore, Bengaluru; sf → San Francisco, Bay Area)
COLLEGES: Include short form, full name, common nicknames
SKILLS: Semantic expansion (frontend → react, vue, angular, javascript, etc.)
COMPANIES: Include subsidiaries and variations

### AND/OR Logic:
- "Stanford AND MIT" → education_logic: "AND", need BOTH schools
- "Stanford OR MIT" → education_logic: "OR", either one
- Default: "OR" for most filters
- Cross-category: ALWAYS "AND"

## OUTPUT FORMAT
Return valid JSON:
{{
    "education": [],
    "education_logic": "OR",
    "education_groups": [],    // Array of {{"canonical": "id", "variations": ["name1", "name2"]}}
    "skills": [],
    "skills_logic": "OR",
    "companies": [],
    "companies_logic": "OR",
    "locations": [],
    "locations_logic": "OR",
    "normalized_query": "",
    "raw_intent": ""
}}

## EXAMPLES

Query: "Stanford and MIT grads"
{{
    "education": ["Stanford", "Stanford University", "MIT", "Massachusetts Institute of Technology"],
    "education_logic": "AND",
    "education_groups": [
        {{"canonical": "stanford", "variations": ["Stanford", "Stanford University", "Stanford GSB"]}},
        {{"canonical": "mit", "variations": ["MIT", "Massachusetts Institute of Technology", "MIT Sloan"]}}
    ],
    "skills": [],
    "skills_logic": "OR",
    "companies": [],
    "companies_logic": "OR",
    "locations": [],
    "locations_logic": "OR",
    "normalized_query": "Stanford and MIT graduates",
    "raw_intent": "People who studied at BOTH Stanford and MIT"
}}

Query: "folks from IISc"
{{
    "education": ["IISc", "Indian Institute of Science", "IISc Bangalore"],
    "education_logic": "OR",
    "education_groups": [
        {{"canonical": "iisc", "variations": ["IISc", "Indian Institute of Science", "IISc Bangalore"]}}
    ],
    "skills": [],
    "skills_logic": "OR",
    "companies": [],
    "companies_logic": "OR",
    "locations": [],
    "locations_logic": "OR",
    "normalized_query": "Indian Institute of Science alumni",
    "raw_intent": "IISc graduates"
}}"""

    user_prompt = f"""Parse and normalize this query: "{query}"

Return ONLY valid JSON, no markdown or explanation."""

    response = await call_llm([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])
    
    try:
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        response = response.strip()
        
        parsed = json.loads(response)
        
        # Cache the result
        if use_cache:
            if len(_query_cache) >= _CACHE_MAX_SIZE:
                # Simple eviction: clear half the cache
                keys_to_remove = list(_query_cache.keys())[:_CACHE_MAX_SIZE // 2]
                for k in keys_to_remove:
                    del _query_cache[k]
            _query_cache[cache_key] = parsed
        
        return parsed
    except json.JSONDecodeError:
        fallback = {
            "education": [],
            "skills": [query],
            "companies": [],
            "locations": [],
            "education_logic": "OR",
            "skills_logic": "OR",
            "companies_logic": "OR",
            "locations_logic": "OR",
            "education_groups": [],
            "normalized_query": query,
            "raw_intent": query
        }
        return fallback


def clear_query_cache():
    """Clear the query normalization cache."""
    global _query_cache
    _query_cache = {}


async def rerank_results(
    query: str,
    candidates: List[Dict[str, Any]],
    parsed_intent: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Use Gemini as a judge to rerank and score candidates.
    Returns candidates with updated scores and explanations.
    """
    if not candidates:
        return []
    
    # Prepare candidate summaries
    candidate_summaries = []
    for i, c in enumerate(candidates[:20]):  # Limit to top 20 for reranking
        summary = f"""
Candidate {i+1} (ID: {c['actor_id']}):
- Name: {c.get('name', 'Unknown')}
- Headline: {c.get('headline', '')}
- Location: {c.get('location', '')}
- Education: {', '.join(c.get('education', []))}
- Companies: {', '.join(c.get('companies', []))}
- Current Role: {c.get('current_role', '')}
"""
        candidate_summaries.append(summary)
    
    system_prompt = """You are a relevance judge for a people search system. 
Score each candidate on how well they match the query intent.

SCORING RULES:
1. Score 0.0-1.0 where 1.0 is perfect match
2. Education queries: candidate MUST have studied at the mentioned school (not just worked there)
3. Skill queries: look for evidence in headline, role titles, and company context
4. Location queries: current location must match
5. Company queries: must have worked at the company
6. Be STRICT about AND logic - if query says "Stanford AND MIT", score 0 if missing either
7. For OR logic, having any one match is sufficient

Output JSON array with scores and brief explanations:
[{"index": 0, "score": 0.85, "reason": "Stanford grad, has ML experience"}, ...]"""

    user_prompt = f"""Query: "{query}"

Parsed Intent:
- Education filter: {parsed_intent.get('education', [])} (logic: {parsed_intent.get('education_logic', 'OR')})
- Skills filter: {parsed_intent.get('skills', [])} (logic: {parsed_intent.get('skills_logic', 'OR')})
- Companies filter: {parsed_intent.get('companies', [])}
- Locations filter: {parsed_intent.get('locations', [])}

Candidates:
{''.join(candidate_summaries)}

Score each candidate. Return ONLY valid JSON array."""

    response = await call_llm([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])
    
    try:
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        response = response.strip()
        
        scores = json.loads(response)
        
        # Apply scores to candidates
        scored_candidates = []
        for score_item in scores:
            idx = score_item.get("index", 0)
            if idx < len(candidates):
                candidate = candidates[idx].copy()
                candidate["score"] = score_item.get("score", 0.5)
                candidate["reason"] = score_item.get("reason", "")
                scored_candidates.append(candidate)
        
        # Sort by score descending
        scored_candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
        return scored_candidates
        
    except (json.JSONDecodeError, KeyError):
        # Return original candidates with default scores
        return candidates


async def evaluate_results(
    query: str,
    results: List[Dict[str, Any]],
    parsed_intent: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Use Gemini as a judge to evaluate retrieval quality and provide feedback.
    """
    if not results:
        return {"score": 0, "feedback": "No results returned", "issues": ["empty_results"]}
    
    result_summaries = []
    for i, r in enumerate(results[:10]):
        summary = f"#{i+1}: {r.get('name', 'Unknown')} - {r.get('headline', '')[:80]} (score: {r.get('score', 0):.2f})"
        result_summaries.append(summary)
    
    system_prompt = """You are evaluating search result quality. Be critical and identify issues.

Check for these problems:
1. EDUCATION LEAKAGE: Did a non-Stanford person appear for "Stanford" query because they worked at a company with Stanford grads?
2. SKILL MISMATCH: Did someone without frontend skills appear for "frontend" query?
3. AND/OR CONFUSION: If query said "A and B", did results include people with only A or only B?
4. LOCATION MISMATCH: Wrong city/country
5. SEMANTIC GAPS: Missing relevant people due to different terminology

Output JSON:
{
    "overall_score": 0-10,
    "precision": 0-1 (what fraction of results are relevant),
    "issues": ["list of specific issues found"],
    "feedback": "detailed feedback for improvement",
    "suggestions": ["specific suggestions to improve retrieval"]
}"""

    user_prompt = f"""Query: "{query}"

Intent: {json.dumps(parsed_intent)}

Top Results:
{chr(10).join(result_summaries)}

Evaluate these results. Return ONLY valid JSON."""

    response = await call_llm([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])
    
    try:
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        response = response.strip()
        
        return json.loads(response)
    except json.JSONDecodeError:
        return {"score": 5, "feedback": "Could not parse evaluation", "issues": []}
