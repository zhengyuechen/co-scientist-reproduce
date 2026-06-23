"""Evolution agent: combine top-two hypotheses into a new evolved hypothesis."""
from __future__ import annotations
import logging

from cosci import run_log
from cosci.agents.base import Results
from cosci.agents.text_utils import clean_title
from cosci.elo import INITIAL_ELO
from cosci.memory import ContextMemory
from cosci.models import AgentName, Hypothesis, Origin, Task, TaskType
from cosci.prompts.reconstructed import EVO_COMBINE
from cosci.prompts.render import render

log = logging.getLogger(__name__)


class EvolutionAgent:
    async def execute(self, task: Task, memory: ContextMemory, llm, cfg) -> Results:
        top_k = cfg.evolution.top_k
        active = memory.active_hypotheses()
        active.sort(key=lambda h: (h.elo_rating if h.elo_rating is not None else INITIAL_ELO), reverse=True)
        top = active[:top_k]

        if len(top) < 2:
            return Results()

        plan = memory.research_plan
        goal = plan.goal if plan else ""
        preferences = "\n".join(plan.preferences) if plan else ""

        prompt = render(
            EVO_COMBINE,
            goal=goal,
            preferences=preferences,
            hypotheses=f"1. {top[0].text}\n\n2. {top[1].text}",
        )

        resp = await llm.complete("evolution", [{"role": "user", "content": prompt}])

        evolved = Hypothesis(
            id=memory.new_id("E"),
            text=resp,
            title=clean_title(resp),
            source_strategy="combine",
            origin=Origin.EVOLVED,
            parent_ids=[top[0].id, top[1].id],
            created_tick=memory.tick,
        )
        memory.add_hypothesis(evolved)
        run_log.emit("evolution_done", tick=memory.tick, child=evolved.id,
                     parents=[top[0].id, top[1].id])

        follow_up = Task(
            agent=AgentName.REFLECTION,
            action=TaskType.REVIEW_HYPOTHESIS,
            target_id=evolved.id,
        )
        return Results(new_hypotheses=[evolved], follow_ups=[follow_up])
