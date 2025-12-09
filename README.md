# Brace Assignment
## Overview

**Given:**
- A dataset of people (`random_actors.json`)
- A set of natural queries (`queries.csv`)

**The system:**
1. Parses queries
2. Converts both queries and actors into vector embeddings
3. Stores actor vectors inside a Pinecone index
4. Performs cosine similarity search
5. Returns top relevant actors
6. Saves results as JSON files


## Design Choices & Reasoning

### Why Embeddings?

Natural language queries are open-ended and vary greatly:
- "someone from blr good with ml"
- "find fintech founders"
- "show me ex-google designers"

Traditional keyword search would fail.
Embedding models (i used gemini-001) convert text into semantic vectors, allowing us to compare meaning instead of exact words.

### Why Pinecone?

Pinecone provides:
- Fast vector similarity search
- Cosine similarity ranking
- Scalable vector storage
- Easy upserts and queries

It fits perfectly with a semantic retrieval system.

## Data Modeling

Each actor is represented by a synthesized text block:

```
Name + Headline + Bio + Location
```

**Example:**
```
"Name: Sarah Chen. Headline: Co-Founder & CEO at CloudFlow AI.
 Bio: Building fintech infrastructure. Location: San Francisco."
```

This ensures the embedding captures:
- Skills
- Industry
- Seniority
- Geography
- Roles
- Experiences

### Vector Representation

- Each actor → one dense embedding vector
- Each query → embedded into the same vector space
- Similarity = cosine distance
- Ranking = top-K highest similarity scores

## Retrieval Pipeline

**The workflow:**

```
Query → Embedding → Pinecone Vector Search → Ranking → JSON Output
```

**Steps:**

### Ingestion
1. Load all actors
2. Vectorize each actor using the embedding model
3. Upsert to Pinecone

### Querying
1. Embed the query
2. Query Pinecone for top-K matches
3. Rank by similarity
4. Output results

### Output Format

```json
{
  "query": "find founders in San Francisco",
  "results": [
    { "actor_id": "sarah_chen", "score": 0.42 },
    { "actor_id": "mike_patel", "score": 0.37 }
  ]
}
```

All outputs are stored locally in `output/`.

## Running the Project

### 1️Install dependencies
```bash
pip install -r requirements.txt
```

### 2️Add environment variables to `.env`
```
OPENROUTER_API_KEY=your_key
PINECONE_API_KEY=your_key
```

### 3️Run the full pipeline (ingest + queries.csv)
```bash
python main.py
```

### 4️Run a custom query
```bash
python main.py "fintech founders working on agents in sf"
```

### 5️Skip ingestion (for rapid testing)
```bash
python main.py --skip-ingest "someone from blr good with ml"
```

### 6️Interactive CLI mode
```bash
python main.py --interactive
```

## Correctness of Retrieval

The system produces highly relevant matches, e.g.

**Query:** "find fintech people"

Results:
- David Thompson — Head of Marketing @ Plaid
- Priya Shah — Growth Marketing @ Chime
- Alex Chen — VP Marketing @ Brex
- Anchit Jain — Fintech + DeFi
- Enrique Ferrao — YC CTO (semantic match)

The retrieval quality is consistent and aligned with query intent.
