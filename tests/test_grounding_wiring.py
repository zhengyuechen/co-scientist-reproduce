import pytest
from cosci.agents.generation import GenerationAgent
from cosci.agents.reflection import ReflectionAgent
from cosci.tools.web_search import Article
from cosci.memory import ContextMemory
from cosci.models import Hypothesis, ResearchPlan, Task, AgentName, TaskType
from cosci.config import load_config
from tests.fake_llm import FakeLLM

class _Backend:
    def __init__(self): self.queries = []
    async def search(self, query, max_results=5):
        self.queries.append(query)
        return [Article(title="Paper1", summary="relevant finding", url="u1", published="2024")]

@pytest.mark.asyncio
async def test_generation_passes_grounded_articles_into_prompt():
    cfg = load_config("config.yaml")
    mem = ContextMemory(research_plan=ResearchPlan(goal="cure X", preferences=["novel"]))
    backend = _Backend()
    captured = {}
    def router(a, m):
        captured["prompt"] = m[-1]["content"]
        return "Hypothesis: grounded idea."
    agent = GenerationAgent(strategies=["literature_review"], grounding=backend)
    await agent.execute(Task(agent=AgentName.GENERATION, action=TaskType.CREATE_INITIAL_HYPOTHESES),
                        mem, FakeLLM(router), cfg)
    assert backend.queries == ["cure X"]                 # searched on the goal
    assert "Paper1" in captured["prompt"]                # article flowed into the rendered prompt

@pytest.mark.asyncio
async def test_reflection_grounds_full_review_on_hypothesis():
    cfg = load_config("config.yaml")
    mem = ContextMemory(research_plan=ResearchPlan(goal="g"))
    mem.add_hypothesis(Hypothesis(id="G1", text="some hypothesis text", title="T", source_strategy="s"))
    backend = _Backend()
    captured = {}
    def router(a, m):
        c = m[-1]["content"]; captured.setdefault("prompts", []).append(c)
        return "verification: verified" if "deep verification" in c else "safety: safe"
    await ReflectionAgent(grounding=backend).execute(
        Task(agent=AgentName.REFLECTION, action=TaskType.REVIEW_HYPOTHESIS, target_id="G1"),
        mem, FakeLLM(router), cfg)
    assert backend.queries and backend.queries[0].startswith("some hypothesis")
    assert any("Paper1" in p for p in captured["prompts"])
    # full review must be marked tool_grounded when grounding returned articles
    full_reviews = [r for r in mem.reviews["G1"] if r.type == "full"]
    assert full_reviews and full_reviews[0].tool_grounded is True

@pytest.mark.asyncio
async def test_generation_no_backend_still_works():
    cfg = load_config("config.yaml")
    mem = ContextMemory(research_plan=ResearchPlan(goal="g", preferences=[]))
    res = await GenerationAgent(strategies=["literature_review"]).execute(
        Task(agent=AgentName.GENERATION, action=TaskType.CREATE_INITIAL_HYPOTHESES),
        mem, FakeLLM(lambda a, m: "Hypothesis: x"), cfg)
    assert len(res.new_hypotheses) == 1     # parametric fallback unchanged
