import pytest
from cosci.agents.meta_review import MetaReviewAgent
from cosci.memory import ContextMemory
from cosci.models import Hypothesis, Review, ResearchPlan, Task, AgentName, TaskType
from cosci.config import load_config
from tests.fake_llm import FakeLLM

@pytest.mark.asyncio
async def test_system_feedback_written():
    cfg = load_config("config.yaml")
    mem = ContextMemory(research_plan=ResearchPlan(goal="g"))
    mem.add_hypothesis(Hypothesis(id="G1", text="t", title="T", source_strategy="s", elo_rating=1200))
    mem.add_review(Review(hypothesis_id="G1", type="full", text="lacks mechanism"))
    res = await MetaReviewAgent().execute(
        Task(agent=AgentName.META_REVIEW, action=TaskType.GENERATE_SYSTEM_FEEDBACK),
        mem, FakeLLM(lambda a, m: "common weakness: vague mechanisms"), cfg)
    assert mem.system_feedback == "common weakness: vague mechanisms"
    assert res.feedback == "common weakness: vague mechanisms"

@pytest.mark.asyncio
async def test_final_overview_returned():
    cfg = load_config("config.yaml")
    mem = ContextMemory(research_plan=ResearchPlan(goal="g"))
    for hid, elo in [("G1", 1300), ("G2", 1250)]:
        mem.add_hypothesis(Hypothesis(id=hid, text=f"text {hid}", title=hid, source_strategy="s", elo_rating=elo))
    res = await MetaReviewAgent().execute(
        Task(agent=AgentName.META_REVIEW, action=TaskType.GENERATE_FINAL_OVERVIEW),
        mem, FakeLLM(lambda a, m: "Research overview: direction A and B."), cfg)
    assert res.overview == "Research overview: direction A and B."

def test_agents_package_exports():
    from cosci.agents import (GenerationAgent, ReflectionAgent, RankingAgent,
                              EvolutionAgent, ProximityAgent, MetaReviewAgent, Results, Agent)
