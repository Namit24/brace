# Bracee People Search

A semantic retrieval system for searching LinkedIn-style profiles using natural language queries.

## Architecture

### The Problem

A naive "one vector per person" approach fails because:
- Career signals (Google, Tesla, AI) dominate embeddings
- Education queries become unreliable (Stanford alumni might show Google employees)
- Skill queries miss semantically equivalent skills ("React" vs "frontend")

### The Solution: Multi-Namespace Semantic Separation + Dynamic Normalization

**1. Separate vectors in isolated namespaces:**

```
┌─────────────────────────────────────────────────────────────────┐
│                        PINECONE INDEX                          │
├─────────────────┬─────────────────┬─────────────┬──────────────┤
│   education     │     skills      │  companies  │   location   │
│   namespace     │    namespace    │  namespace  │   namespace  │
├─────────────────┼─────────────────┼─────────────┼──────────────┤
│ "MIT, CS"       │ "ML, deep       │ "Google,    │ "San         │
│ "Stanford, MBA" │  learning, NLP" │  Amazon"    │  Francisco"  │
│                 │ "frontend,      │             │ "Bangalore"  │
│                 │  React, Vue"    │             │              │
└─────────────────┴─────────────────┴─────────────┴──────────────┘
```

**2. Dynamic Query Normalization (no static JSON files!):**

Instead of maintaining hardcoded alias files, Gemini dynamically expands queries:

```
User Query: "folks who studied at IITB"
     │
     ▼ Gemini Normalization
{
  "education": ["IIT Bombay", "Indian Institute of Technology Bombay", "IITB"],
  "normalized_query": "IIT Bombay alumni Indian Institute of Technology Bombay graduates"
}
```

```
User Query: "frontend devs in sf"
     │
     ▼ Gemini Normalization
{
  "skills": ["frontend", "react", "vue", "angular", "javascript", "typescript", "ui engineer"],
  "locations": ["San Francisco", "Bay Area"],
  "normalized_query": "frontend developers in San Francisco"
}
```

This approach:
- **Scales automatically** - no need to add new aliases to JSON files
- **Handles edge cases** - Gemini understands context and variations
- **Works with any data** - new schools/skills/companies work out of the box

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys:
# OPENROUTER_API_KEY=your_key
# PINECONE_API_KEY=your_key
```

## Usage

### Quick Start
```bash
# Ingest data into Pinecone
python main.py

# Ingest and start interactive query mode
python main.py --interactive

# Interactive mode without re-ingesting
python main.py --interactive --skip-ingest

# Run a single query
python main.py --query "frontend devs from Bangalore"

# Reset index and re-ingest
python main.py --reset
```

### CLI Options
```
python main.py [OPTIONS]

Options:
  -i, --interactive    Start interactive query mode
  -s, --skip-ingest    Skip data ingestion (use existing index)
  -q, --query TEXT     Run a single query and exit
  --actors PATH        Path to actors JSON (default: data/random_actors.json)
  --reset              Reset Pinecone namespaces before ingestion
  --debug              Enable debug output
```

## Output Format

```json
{
  "query": "frontend devs from Bangalore",
  "results": [
    { "actor_id": "arjunmenon-ml-blr", "score": 0.89 },
    { "actor_id": "priya-frontend-blr", "score": 0.82 }
  ]
}
```

## Why This Design Avoids Failure Modes

| Failure Mode | How We Avoid It |
|--------------|-----------------|
| Career dominates education | Separate namespaces - education query only hits education vectors |
| "MIT" ≠ "Massachusetts Institute of Technology" | Dynamic LLM expansion (no static aliases needed) |
| "frontend" misses React developers | Gemini expands to all related technologies |
| "Stanford AND MIT" returns union | Explicit intersection logic after retrieval |
| Semantic drift in results | LLM reranker as final quality gate |
| New schools/skills not in config | LLM handles any entity dynamically |

## Project Structure

```
bracee/
├── main.py                     # Main CLI entry point
├── src/
│   ├── aliases.py              # Structured alias store (schools, locations, skills)
│   ├── embeddings.py           # Gemini embedding client (OpenRouter)
│   ├── llm.py                  # Query normalization with caching & reranking
│   ├── data_processing.py      # Actor → chunks transformation
│   ├── pinecone_db.py          # Vector DB operations
│   └── retriever.py            # Main search orchestration
├── scripts/
│   ├── ingest.py               # Standalone ingestion script
│   └── run_queries.py          # Batch query execution
├── data/
│   ├── random_actors.json      # Input profiles
│   ├── queries.csv             # Test queries
│   └── profiles_cache.json     # Generated profile cache
└── output/
    ├── results.json            # Query results
    └── evaluations.json        # LLM quality evaluations
```
