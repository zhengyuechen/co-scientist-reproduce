"""Meta-review agent: synthesizes system-wide feedback and final research overview."""
from __future__ import annotations

from cosci.agents.base import Results
from cosci.memory import ContextMemory
from cosci.models import Task, TaskType
from cosci.prompts.reconstructed import META_SYSTEM_FEEDBACK, META_RESEARCH_OVERVIEW
from cosci.prompts.render import render


class MetaReviewAgent:
    async def execute(self, task: Task, memory: ContextMemory, llm, cfg) -> Results:
        goal = memory.research_plan.goal

        if task.action == TaskType.GENERATE_SYSTEM_FEEDBACK:
            all_texts = []
            for reviews in memory.reviews.values():
                for review in reviews:
                    all_texts.append(review.text)
            for match in memory.tournament:
                all_texts.append(match.transcript)
            reviews_block = "\n\n".join(all_texts)
            rendered = render(META_SYSTEM_FEEDBACK, goal=goal, reviews=reviews_block)
            response = await llm.complete("meta_review", [{"role": "user", "content": rendered}])
            memory.system_feedback = response
            return Results(feedback=response)

        if task.action == TaskType.GENERATE_FINAL_OVERVIEW:
            top_n = cfg.overview.top_n
            active = sorted(
                memory.active_hypotheses(),
                key=lambda h: h.elo_rating if h.elo_rating is not None else 0.0,
                reverse=True,
            )[:top_n]
            hypotheses_block = "\n".join(f"{i + 1}. {h.text}" for i, h in enumerate(active))
            rendered = render(META_RESEARCH_OVERVIEW, goal=goal, hypotheses=hypotheses_block)
            response = await llm.complete("meta_review", [{"role": "user", "content": rendered}])
            return Results(overview=response)

        return Results()
