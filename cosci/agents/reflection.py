"""Reflection agent: runs full and deep-verification reviews on a hypothesis."""
from __future__ import annotations

from cosci import run_log
from cosci.agents.base import Results, parse_label
from cosci.memory import ContextMemory
from cosci.models import AgentName, Review, Safety, Task, TaskType
from cosci.prompts.reconstructed import REFLECT_DEEP_VERIFICATION, REFLECT_FULL
from cosci.prompts.render import render
from cosci.tools.web_search import backend_label, safe_search


class ReflectionAgent:
    def __init__(self, grounding=None) -> None:
        self.grounding = grounding

    async def execute(self, task: Task, memory: ContextMemory, llm, cfg) -> Results:
        hid = task.target_id
        hypothesis = memory.get(hid)
        run_log.emit("reflection_started", tick=memory.tick, hypothesis_id=hid)

        # --- Full review --- (grounding failures degrade to parametric, never crash the run)
        grounding_query = f"{memory.research_plan.goal} {hypothesis.title}".strip()
        if self.grounding is not None:
            run_log.emit("grounding_search", tick=memory.tick, query=grounding_query,
                         backend=backend_label(self.grounding))
        articles_block = await safe_search(self.grounding, grounding_query)
        if self.grounding is not None:
            run_log.emit("grounding_result", tick=memory.tick, articles=articles_block.count("URL:"))
        full_prompt = render(
            REFLECT_FULL,
            goal=memory.research_plan.goal,
            hypothesis=hypothesis.text,
            articles_with_reasoning=articles_block,
        )
        full_response = await llm.complete("reflection", [{"role": "user", "content": full_prompt}])
        if articles_block:
            full_response = (
                f"{full_response.rstrip()}\n\n"
                "Grounding sources provided to reviewer:\n"
                f"{articles_block}"
            )

        safety_verdict = parse_label(full_response, "safety")
        if safety_verdict == "unsafe":
            review_safety = Safety.UNSAFE
            hypothesis.safety = Safety.UNSAFE
            safety_line = next(
                (ln for ln in full_response.splitlines() if "safety" in ln.lower()),
                full_response,
            )
            hypothesis.safety_reason = safety_line
        else:
            review_safety = Safety.SAFE
            hypothesis.safety = Safety.SAFE

        run_log.emit("reflection_done", tick=memory.tick, hypothesis_id=hid,
                     grounded=bool(articles_block), safety=review_safety)

        full_review = Review(
            hypothesis_id=hid,
            type="full",
            text=full_response,
            tool_grounded=bool(articles_block),
            safety=review_safety,
        )

        # --- Deep verification review ---
        deep_prompt = render(
            REFLECT_DEEP_VERIFICATION,
            hypothesis=hypothesis.text,
        )
        deep_response = await llm.complete("reflection", [{"role": "user", "content": deep_prompt}])

        deep_review = Review(
            hypothesis_id=hid,
            type="deep_verification",
            text=deep_response,
            tool_grounded=False,
        )

        memory.add_review(full_review)
        memory.add_review(deep_review)

        follow_up = Task(
            agent=AgentName.RANKING,
            action=TaskType.ADD_TO_TOURNAMENT,
            target_id=hid,
        )

        return Results(reviews=[full_review, deep_review], follow_ups=[follow_up])
