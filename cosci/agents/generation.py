"""Generation agent: produces new hypotheses from research strategies."""
from __future__ import annotations

from cosci.agents.base import Results
from cosci.memory import ContextMemory
from cosci.models import AgentName, Hypothesis, Origin, Task, TaskType
from cosci.prompts.reconstructed import GEN_ITERATIVE_ASSUMPTIONS, GEN_RESEARCH_EXPANSION
from cosci.prompts.render import assemble_instructions, render
from cosci.prompts.verbatim import GEN_DEBATE, GEN_LITERATURE
from cosci.tools.web_search import format_articles

_STRATEGY_PROMPT = {
    "literature_review": GEN_LITERATURE,
    "scientific_debate": GEN_DEBATE,
    "iterative_assumptions": GEN_ITERATIVE_ASSUMPTIONS,
    "research_expansion": GEN_RESEARCH_EXPANSION,
}

_DEFAULT_STRATEGIES = ["literature_review", "scientific_debate"]


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
            if strategy == "literature_review" and self.grounding is not None:
                articles = await self.grounding.search(goal, max_results=5)
                articles_block = format_articles(articles)
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

            text = response
            if strategy == "scientific_debate" and "HYPOTHESIS" in response:
                text = response.split("HYPOTHESIS")[-1]

            text = text.strip()
            first_line = next((ln for ln in text.splitlines() if ln.strip()), text)
            title = first_line.strip()[:80]

            h = Hypothesis(
                id=memory.new_id("G"),
                title=title,
                text=text,
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
