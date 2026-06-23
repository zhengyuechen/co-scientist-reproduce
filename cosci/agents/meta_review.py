"""Meta-review agent: synthesizes system-wide feedback and final research overview."""
from __future__ import annotations

from cosci.agents.base import Results, passes_novelty_gate
from cosci.memory import ContextMemory
from cosci.models import Task, TaskType
from cosci.prompts.reconstructed import META_SYSTEM_FEEDBACK, META_RESEARCH_OVERVIEW
from cosci.prompts.render import render


def _latest_full_review(memory: ContextMemory, hid: str) -> str:
    fulls = [r for r in memory.reviews.get(hid, []) if r.type == "full"]
    return fulls[-1].text if fulls else ""


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
            active = sorted(
                memory.active_hypotheses(),
                key=lambda h: h.elo_rating if h.elo_rating is not None else 0.0,
                reverse=True,
            )
            # Defense-in-depth: low-novelty hypotheses are pruned at reflection (active=False),
            # but re-apply the gate here so an active-but-condemned one can never slip in.
            kept = [h for h in active if passes_novelty_gate(h, min_novelty)]
            chosen = kept[: cfg.overview.top_n]
            # Excluded from the synthesis as restatements: pruned upstream + any caught here.
            pruned_upstream = sum(1 for h in memory.hypotheses.values() if not h.active and h.pruned_reason)
            excluded = pruned_upstream + (len(active) - len(kept))

            # Tier 1: each surviving hypothesis carries its review into the synthesis, so the
            # overview can name the closest existing model per direction instead of laundering it.
            items = []
            for i, h in enumerate(chosen):
                nov = f" [novelty {int(h.novelty)}/10]" if h.novelty is not None else ""
                item = f"{i + 1}. {h.text}{nov}"
                review = _latest_full_review(memory, h.id)
                if review:
                    item += f"\n   Review (identifies the closest existing model): {review[:1200]}"
                items.append(item)
            body = "\n\n".join(items)

            if excluded:
                note = (
                    f"NOTE: {excluded} candidate hypotheses were pruned as low-novelty restatements "
                    f"of existing models and excluded from this synthesis. "
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
