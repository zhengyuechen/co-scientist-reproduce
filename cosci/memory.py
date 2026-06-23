"""ContextMemory == SM 'SharedMemory'. JSON-snapshot for restartability.
Concurrency note: callers serialize mutations via an asyncio.Lock in the engine's scheduler;
these methods are plain sync."""
from __future__ import annotations
from pydantic import BaseModel
from cosci.models import Hypothesis, Review, MatchResult, ResearchPlan, Safety

class ContextMemory(BaseModel):
    research_plan: ResearchPlan | None = None
    hypotheses: dict[str, Hypothesis] = {}
    reviews: dict[str, list[Review]] = {}
    tournament: list[MatchResult] = []
    proximity: dict[str, list[dict]] = {}
    system_feedback: str = ""
    tick: int = 0
    tokens_spent: int = 0
    id_counter: int = 0

    def new_id(self, prefix: str) -> str:
        self.id_counter += 1
        return f"{prefix}{self.id_counter}"

    def add_hypothesis(self, h: Hypothesis) -> None:
        self.hypotheses[h.id] = h

    def get(self, hid: str) -> Hypothesis | None:
        return self.hypotheses.get(hid)

    def active_hypotheses(self) -> list[Hypothesis]:
        return [h for h in self.hypotheses.values()
                if h.active and h.safety != Safety.UNSAFE]

    def add_review(self, r: Review) -> None:
        self.reviews.setdefault(r.hypothesis_id, []).append(r)

    def record_match(self, m: MatchResult) -> None:
        self.tournament.append(m)

    def save_snapshot(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load_snapshot(cls, path: str) -> "ContextMemory":
        with open(path) as f:
            return cls.model_validate_json(f.read())
