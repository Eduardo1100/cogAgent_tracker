"""Semantic retrieval utilities for episodic and concept memory.

Uses cosine similarity over sentence-transformer embeddings to rank
episodes and knowledge concepts by relevance to the current task,
replacing random sampling in retrieve_memory() and generate_initial_message().

Embeddings are cached per-call by list length so they are only recomputed
when new items are added.
"""

import json

import numpy as np

from src.agent.helpers import sentence_transformer_model


def _embed(texts: list[str]) -> np.ndarray:
    """Return L2-normalised float32 embeddings for cosine similarity via dot product."""
    embs = sentence_transformer_model.encode(texts, convert_to_numpy=True).astype(
        np.float32
    )
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return embs / norms


def _episode_to_text(episode: dict) -> str:
    memory = episode.get("memory", [])
    if isinstance(memory, list):
        snippet = " ".join(
            m if isinstance(m, str) else json.dumps(m) for m in memory[:5]
        )
    else:
        snippet = str(memory)
    return f"outcome={episode.get('task_outcome', '')} {snippet}"


def retrieve_relevant_episodes(
    episodes: list[dict],
    query: str,
    k: int = 5,
    cache: dict | None = None,
) -> list[dict]:
    """Return up to k episodes most semantically similar to query.

    Results are returned sorted chronologically by episode_number so the
    agent sees them in the order they were experienced.

    cache: optional mutable dict used to skip re-embedding an unchanged list.
    """
    if not episodes:
        return []

    k = min(k, len(episodes))
    n = len(episodes)

    if cache is not None and cache.get("n") == n:
        episode_embs = cache["embs"]
    else:
        texts = [_episode_to_text(ep) for ep in episodes]
        episode_embs = _embed(texts)
        if cache is not None:
            cache["n"] = n
            cache["embs"] = episode_embs

    query_emb = _embed([query])[0]
    scores = episode_embs @ query_emb
    top_indices = np.argsort(scores)[::-1][:k]
    return [episodes[i] for i in sorted(top_indices)]


def retrieve_relevant_concepts(
    knowledge: list[str],
    query: str,
    k: int = 5,
    cache: dict | None = None,
) -> list[str]:
    """Return up to k knowledge cluster entries most relevant to query.

    cache: optional mutable dict used to skip re-embedding an unchanged list.
    """
    if not knowledge:
        return knowledge

    k = min(k, len(knowledge))
    n = len(knowledge)

    if cache is not None and cache.get("n") == n:
        concept_embs = cache["embs"]
    else:
        concept_embs = _embed(knowledge)
        if cache is not None:
            cache["n"] = n
            cache["embs"] = concept_embs

    query_emb = _embed([query])[0]
    scores = concept_embs @ query_emb
    top_indices = np.argsort(scores)[::-1][:k]
    return [knowledge[i] for i in top_indices]
