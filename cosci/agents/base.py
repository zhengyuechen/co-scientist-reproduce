"""Shared agent contract: Results payload, Agent protocol, and a verdict-line parser."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Protocol
from cosci.models import Hypothesis, Review, MatchResult, Task
from cosci.memory import ContextMemory
from cosci.config import Config
from cosci.llm import LLMClient

@dataclass
class Results:
    new_hypotheses: list[Hypothesis] = field(default_factory=list)
    reviews: list[Review] = field(default_factory=list)
    match: MatchResult | None = None
    follow_ups: list[Task] = field(default_factory=list)
    overview: str | None = None
    feedback: str | None = None

class Agent(Protocol):
    async def execute(self, task: Task, memory: ContextMemory,
                      llm: LLMClient, cfg: Config) -> Results: ...

def parse_label(text: str, *labels: str) -> str | None:
    """Return the value of the LAST `label: <value>` marker (case-insensitive), or None.
    Strips surrounding <> and trailing prose after the value token-run."""
    best = None
    for label in labels:
        for m in re.finditer(rf"\b{re.escape(label)}\s*:\s*<?\s*([A-Za-z0-9 _\-]+?)\s*>?(?:\s|$|[.,;])",
                             text, re.IGNORECASE):
            best = m.group(1).strip().lower()
    return best
