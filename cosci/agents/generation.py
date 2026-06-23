"""Generation agent: produces new hypotheses from research strategies."""
from __future__ import annotations

from cosci.agents.base import Results
from cosci.agents.text_utils import clean_title, split_atomic_hypotheses
from cosci.memory import ContextMemory
from cosci.models import AgentName, Hypothesis, Origin, Task, TaskType
from cosci.prompts.reconstructed import GEN_ITERATIVE_ASSUMPTIONS, GEN_RESEARCH_EXPANSION
from cosci.prompts.render import assemble_instructions, render
from cosci.prompts.verbatim import GEN_DEBATE, GEN_LITERATURE
from cosci.tools.web_search import safe_search

_STRATEGY_PROMPT = {
    "literature_review": GEN_LITERATURE,
    "scientific_debate": GEN_DEBATE,
    "iterative_assumptions": GEN_ITERATIVE_ASSUMPTIONS,
    "research_expansion": GEN_RESEARCH_EXPANSION,
}

_DEFAULT_STRATEGIES = ["literature_review", "scientific_debate", "iterative_assumptions", "research_expansion"]


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
            if strategy == "literature_review":
                articles_block = await safe_search(self.grounding, goal)
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
            # strategy may instead converge on a single finalized idea after a bare
            # "HYPOTHESIS" marker — honor that only when it is not a numbered bundle.
            raw = response.strip()
            chunks = split_atomic_hypotheses(raw)
            if len(chunks) == 1 and strategy == "scientific_debate" and "HYPOTHESIS" in raw:
                converged = raw.split("HYPOTHESIS")[-1].strip()
                if converged:
                    chunks = [converged]

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

        return Results(new_hypotheses=new_hypotheses, follow_ups=follow_ups)
