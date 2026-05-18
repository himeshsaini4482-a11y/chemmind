import os
from typing import Optional
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client


SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
NOVELTY_THRESHOLD = 0.15
PRUNE_SCORE_THRESHOLD = 0.2
PRUNE_AGE_DAYS = 30

_model = SentenceTransformer(EMBEDDING_MODEL)
_supabase: Optional[Client] = None


def _get_supabase() -> Client:
    global _supabase
    if _supabase is None:
        _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase


def _embed(text: str) -> list[float]:
    return _model.encode(text).tolist()


def store_memory(
    text: str,
    material_system: Optional[str] = None,
    simulation_software: Optional[str] = None,
    memory_type: str = "general",
) -> str:
    embedding = _embed(text)
    novelty = compute_novelty(text)

    row = {
        "content": text,
        "embedding": embedding,
        "novelty_score": novelty,
        "material_system": material_system,
        "simulation_software": simulation_software,
        "memory_type": memory_type,
        "created_at": datetime.utcnow().isoformat(),
    }

    result = _get_supabase().table("memories").insert(row).execute()
    return result.data[0]["id"]


def retrieve_memories(
    query: str,
    top_k: int = 5,
    material_system: Optional[str] = None,
    simulation_software: Optional[str] = None,
    memory_type: Optional[str] = None,
    min_novelty: float = 0.0,
) -> list[dict]:
    embedding = _embed(query)
    embedding_str = str(embedding).replace(" ", "")

    sql = """
        SELECT id, content, material_system, simulation_software,
               memory_type, novelty_score, created_at,
               1 - (embedding <=> $1::vector) AS similarity
        FROM memories
        WHERE novelty_score >= $2
    """
    params: list = [embedding_str, min_novelty]

    if material_system:
        params.append(material_system)
        sql += f" AND material_system = ${len(params)}"
    if simulation_software:
        params.append(simulation_software)
        sql += f" AND simulation_software = ${len(params)}"
    if memory_type:
        params.append(memory_type)
        sql += f" AND memory_type = ${len(params)}"

    sql += f" ORDER BY embedding <=> $1::vector ASC LIMIT {top_k}"

    result = _get_supabase().rpc("execute_sql", {"query": sql, "params": params}).execute()
    return result.data


def compute_novelty(text: str) -> float:
    result = _get_supabase().table("memories").select("embedding").execute()

    if not result.data:
        return 1.0

    stored = [r["embedding"] for r in result.data]
    query_vec = np.array(_embed(text))
    stored_matrix = np.array(stored)

    norms = np.linalg.norm(stored_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    cosine_sim = (stored_matrix @ query_vec) / (norms.squeeze() * np.linalg.norm(query_vec))

    max_sim = cosine_sim.max()
    return round(1.0 - max_sim, 4)


def prune_memories(
    score_threshold: float = PRUNE_SCORE_THRESHOLD,
    age_days: int = PRUNE_AGE_DAYS,
) -> int:
    cutoff = (datetime.utcnow().timestamp() - age_days * 86400) * 1000

    result = (
        _get_supabase()
        .table("memories")
        .delete()
        .lt("novelty_score", score_threshold)
        .lt("created_at", datetime.utcfromtimestamp(cutoff / 1000).isoformat())
        .execute()
    )

    return len(result.data) if result.data else 0
