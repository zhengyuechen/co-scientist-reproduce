"""Meta-review agent: synthesizes system-wide feedback and final research overview."""
from __future__ import annotations

from cosci.agents.base import Results
from cosci.memory import ContextMemory
from cosci.models import Task, TaskType
from cosci.prompts.reconstructed import META_SYSTEM_FEEDBACK, META_RESEARCH_OVERVIEW
from cosci.prompts.render import render


def passes_novelty_gate(h, min_novelty: float) -> bool:
    """A hypothesis reaches the overview unless its own review verdicts condemn it.

    Fails open: a hypothesis with no parsed novelty score is kept (we only drop on a
    judgment we actually have), so runs predating the parsed verdicts are unaffected.
    """
    if h.verification == "invalidated":
        return False
    if h.novelty is not None and h.novelty < min_novelty:
        return False
    return True


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
            min_novelty = cfg.overview.min_novelty
            ranked = sorted(
                memory.active_hypotheses(),
                key=lambda h: h.elo_rating if h.elo_rating is not None else 0.0,
                reverse=True,
            )
            # Gate: only hypotheses whose own reviews did not condemn them as restatements
            # (low novelty) or invalidated reach the synthesis. The criticism the model
            # already wrote now has somewhere to land.
            kept = [h for h in ranked if passes_novelty_gate(h, min_novelty)]
            dropped = len(ranked) - len(kept)
            chosen = kept[: cfg.overview.top_n]

            body = "\n".join(f"{i + 1}. {h.text}" for i, h in enumerate(chosen))
            if dropped:
                note = (
                    f"NOTE: {dropped} of {len(ranked)} candidate hypotheses were excluded as "
                    f"low-novelty (novelty below {min_novelty}/10 or verification invalidated) — "
                    f"their own reviews judged them restatements of existing models. "
                )
                if not chosen:
                    note += (
                        "No candidate cleared the novelty bar. Report plainly that the space "
                        "appears well-trodden and that no novel direction emerged, rather than "
                        "synthesizing one. "
                    )
                body = note + "\n\n" + body
            rendered = render(META_RESEARCH_OVERVIEW, goal=goal, hypotheses=body)
            response = await llm.complete("meta_review", [{"role": "user", "content": rendered}])
            return Results(overview=response)

        return Results()
