"""Generation agent: produces new hypotheses from research strategies."""
from __future__ import annotations

import re

from cosci import run_log
from cosci.agents.base import Results
from cosci.agents.text_utils import clean_title, split_atomic_hypotheses
from cosci.memory import ContextMemory
from cosci.models import AgentName, Hypothesis, Origin, Task, TaskType
from cosci.prompts.reconstructed import GEN_ITERATIVE_ASSUMPTIONS, GEN_RESEARCH_EXPANSION
from cosci.prompts.render import assemble_instructions, render
from cosci.prompts.verbatim import GEN_DEBATE, GEN_LITERATURE
from cosci.tools.web_search import backend_label, safe_search

_STRATEGY_PROMPT = {
    "literature_review": GEN_LITERATURE,
    "scientific_debate": GEN_DEBATE,
    "iterative_assumptions": GEN_ITERATIVE_ASSUMPTIONS,
    "research_expansion": GEN_RESEARCH_EXPANSION,
}

_DEFAULT_STRATEGIES = ["literature_review", "scientific_debate", "iterative_assumptions", "research_expansion"]

# SN9 debate responses terminate with a bare "HYPOTHESIS" (all caps) before the
# finalized idea. The negative lookahead excludes a numbered "HYPOTHESIS 2:" proposal
# label, which is a deliberation candidate, not the termination marker.
_DEBATE_TERMINATION_RE = re.compile(r"HYPOTHESIS\b(?!\s*#?\s*\d)")


def _debate_chunks(raw: str) -> list[str]:
    """Atomic hypotheses from a scientific-debate response.

    Prefer the finalized idea after the last bare HYPOTHESIS marker, splitting it
    only if that idea is itself a numbered bundle. With no marker (the model gave
    its opening contribution of several numbered proposals instead of converging),
    split the whole response so each proposal becomes its own hypothesis.
    """
    marks = list(_DEBATE_TERMINATION_RE.finditer(raw))
    if marks:
        converged = raw[marks[-1].end():].strip().lstrip(":-–— ").strip()
        if converged:
            return split_atomic_hypotheses(converged)
    return split_atomic_hypotheses(raw)


class GenerationAgent:
    def __init__(self, strategies: list[str] | None = None, grounding=None) -> None:
        self.strategies = strategies if strategies is not None else _DEFAULT_STRATEGIES
        self.grounding = grounding

    async def execute(self, task: Task, memory: ContextMemory, llm, cfg) -> Results:
        goal = memory.research_plan.goal
        preferences = "\n".join(memory.research_plan.preferences)
        instructions = assemble_instructions(memory)

        new_hypotheses: list[Hypothesis] = []
        follow_ups: list[Task] = []

        for strategy in self.strategies:
            template = _STRATEGY_PROMPT[strategy]
            run_log.emit("generation_started", tick=memory.tick, strategy=strategy)
            if strategy == "literature_review":
                if self.grounding is not None:
                    run_log.emit("grounding_search", tick=memory.tick, query=goal,
                                 backend=backend_label(self.grounding))
                articles_block = await safe_search(self.grounding, goal)
                if self.grounding is not None:
                    run_log.emit("grounding_result", tick=memory.tick,
                                 articles=articles_block.count("URL:"))
            else:
                articles_block = ""
            rendered = render(
                template,
                goal=goal,
                preferences=preferences,
                instructions=instructions,
                idea_attributes="novel",
                source_hypothesis="",
                articles_with_reasoning=articles_block,
                reviews_overview="",
                transcript="",
                research_overview="",
            )
            response = await llm.complete("generation", [{"role": "user", "content": rendered}])

            # Split a response that bundles several numbered proposals into atomic
            # hypotheses so each is reviewed, ranked, and cited on its own. The debate
            # strategy first resolves its "HYPOTHESIS" convergence marker (if any),
            # so a deliberation that weighed several candidates collapses to the final
            # idea rather than being split into the candidates.
            raw = response.strip()
            if strategy == "scientific_debate":
                chunks = _debate_chunks(raw)
            else:
                chunks = split_atomic_hypotheses(raw)

            for chunk in chunks:
                h = Hypothesis(
                    id=memory.new_id("G"),
                    title=clean_title(chunk),
                    text=chunk,
                    source_strategy=strategy,
                    origin=Origin.GENERATED,
                    created_tick=memory.tick,
                )
                memory.add_hypothesis(h)
                new_hypotheses.append(h)
                follow_ups.append(
                    Task(agent=AgentName.REFLECTION, action=TaskType.REVIEW_HYPOTHESIS, target_id=h.id)
                )
            run_log.emit("generation_done", tick=memory.tick, strategy=strategy, hypotheses=len(chunks))

        return Results(new_hypotheses=new_hypotheses, follow_ups=follow_ups)
