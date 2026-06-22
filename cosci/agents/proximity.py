"""Proximity agent: builds an embedding similarity graph over active hypotheses."""
from __future__ import annotations
import math

from cosci.agents.base import Results
from cosci.memory import ContextMemory
from cosci.models import Task


def _cosine(u, v) -> float:
    u = list(map(float, u))
    v = list(map(float, v))
    dot = sum(a * b for a, b in zip(u, v))
    norm_u = math.sqrt(sum(a * a for a in u))
    norm_v = math.sqrt(sum(b * b for b in v))
    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0
    return dot / (norm_u * norm_v)


class ProximityAgent:
    def __init__(self, encoder=None):
        self.encoder = encoder

    async def execute(self, task: Task, memory: ContextMemory, llm, cfg) -> Results:
        active = memory.active_hypotheses()
        if len(active) < 2:
            return Results()

        if self.encoder is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
            self.encoder = SentenceTransformer(cfg.proximity.model)

        vectors = self.encoder.encode([h.text for h in active])

        for i, hi in enumerate(active):
            memory.proximity[hi.id] = [
                {"other_id": active[j].id, "similarity": _cosine(vectors[i], vectors[j])}
                for j in range(len(active)) if j != i
            ]

        return Results()
