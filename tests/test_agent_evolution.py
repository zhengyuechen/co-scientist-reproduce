import pytest
from cosci.agents.evolution import EvolutionAgent
from cosci.elo import INITIAL_ELO
from cosci.memory import ContextMemory
from cosci.models import Hypothesis, ResearchPlan, Task, AgentName, TaskType, Origin
from cosci.config import load_config
from tests.fake_llm import FakeLLM

def _mk(mem, hid, elo):
    h = Hypothesis(id=hid, text=f"text {hid}", title=hid, source_strategy="s", elo_rating=elo)
    mem.add_hypothesis(h); return h

@pytest.mark.asyncio
async def test_evolution_combines_top_two():
    cfg = load_config("config.yaml")
    mem = ContextMemory(research_plan=ResearchPlan(goal="g", preferences=["novel"]))
    _mk(mem, "G1", 1300); _mk(mem, "G2", 1250); _mk(mem, "G3", 1200)
    res = await EvolutionAgent().execute(
        Task(agent=AgentName.EVOLUTION, action=TaskType.EVOLVE_TOP),
        mem, FakeLLM(lambda a, m: "Combined hypothesis text here."), cfg)
    assert len(res.new_hypotheses) == 1
    ev = res.new_hypotheses[0]
    assert ev.origin == Origin.EVOLVED and ev.source_strategy == "combine"
    assert ev.parent_ids == ["G1", "G2"]
    assert mem.get(ev.id) is not None
    assert res.follow_ups[0].action == TaskType.REVIEW_HYPOTHESIS

@pytest.mark.asyncio
async def test_evolution_noop_with_fewer_than_two():
    cfg = load_config("config.yaml")
    mem = ContextMemory(research_plan=ResearchPlan(goal="g"))
    _mk(mem, "G1", 1200)
    res = await EvolutionAgent().execute(
        Task(agent=AgentName.EVOLUTION, action=TaskType.EVOLVE_TOP),
        mem, FakeLLM(lambda a, m: "x"), cfg)
    assert res.new_hypotheses == []
