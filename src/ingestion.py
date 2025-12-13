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
        profile_parts = [
            f"Name: {actor.profile.name}.",
            f"Headline: {actor.profile.headline or ''}.",
            f"Bio: {actor.profile.bio or ''}.",
            f"Location: {actor.profile.location or ''}.",
        ]

        work_parts = []
        for job in actor.professional.workexperience:
            work_parts.append(f"{job.title} at {job.companyname}. {job.description or ''}")

        edu_parts = []
        for edu in actor.professional.education:
            edu_parts.append(f"Studied {edu.degree or ''} {edu.fieldofstudy or ''} at {edu.school}.")

        full_text = " ".join(profile_parts + work_parts + edu_parts)

        vec = get_embedding(full_text)
        if not vec:
            continue

        vectors.append(
            VectorRecord(
                id=actor.unique_id,
                values=vec,
                metadata={
                    "name": actor.profile.name,
                    "education": " ".join(edu_parts),
                    "context": full_text[:1000],
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
