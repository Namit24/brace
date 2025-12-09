import json
import time
from typing import List

from tqdm import tqdm

from src.models import Actor, VectorRecord
from src.utils import get_embedding, get_pinecone_index
from src.config import Config
from dotenv import load_dotenv
load_dotenv()



def load_actors() -> List[Actor]:
    try:
        with open(Config.DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [Actor.model_validate(item) for item in data]
    except Exception as e:
        print(f"Error loading data: {e}")
        return []


def vectorize_actors(actors: List[Actor]) -> List[VectorRecord]:
    vectors: List[VectorRecord] = []
    print(f"Vectorizing {len(actors)} actors")
    for actor in tqdm(actors):
        parts = [
            f"Name: {actor.profile.name}.",
            f"Headline: {actor.profile.headline or ''}.",
            f"Bio: {actor.profile.bio or ''}.",
            f"Location: {actor.profile.location or ''}.",
        ]
        for job in actor.professional.workexperience:
            parts.append(f"Role: {job.title} at {job.companyname}.")
            if job.description:
                parts.append(f"Details: {job.description}.")
        for edu in actor.professional.education:
            degree = edu.degree or ""
            field = edu.fieldofstudy or ""
            parts.append(f"Studied {degree} {field} at {edu.school}.")
        text = " ".join(parts)
        vec = None
        for _ in range(3):
            vec = get_embedding(text)
            if vec:
                break
            time.sleep(1)
        if not vec:
            print(f"Failed to embed: {actor.profile.name}")
            continue
        vectors.append(
            VectorRecord(
                id=actor.unique_id,
                values=vec,
                metadata={
                    "name": actor.profile.name,
                    "headline": actor.profile.headline or "",
                    "context": text[:1000],
                },
            )
        )
    return vectors


def upload_vectors(vectors: List[VectorRecord]) -> None:
    if not vectors:
        print("No vectors to upload")
        return
    index = get_pinecone_index()
    batch_size = 50
    print(f"Upserting {len(vectors)} vectors")
    for i in tqdm(range(0, len(vectors), batch_size)):
        batch = vectors[i : i + batch_size]
        payload = [v.model_dump() for v in batch]
        for attempt in range(3):
            try:
                index.upsert(vectors=payload, namespace=Config.NAMESPACE)
                break
            except Exception as e:
                print(f"Batch {i // batch_size + 1} failed (attempt {attempt + 1}): {e}")
                time.sleep(2)


if __name__ == "__main__":
    actors = load_actors()
    vectors = vectorize_actors(actors)
    upload_vectors(vectors)
