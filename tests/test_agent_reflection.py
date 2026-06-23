import pytest
from cosci.agents.reflection import ReflectionAgent
from cosci.memory import ContextMemory
from cosci.models import Hypothesis, ResearchPlan, Task, AgentName, TaskType, Safety
from cosci.config import load_config
from tests.fake_llm import FakeLLM

def _safe_router(agent, messages):
    c = messages[-1]["content"]
    if "deep verification" in c: return "assumptions ok\nverification: verified"
    return "novel and correct\nsafety: safe"

def _unsafe_router(agent, messages):
    c = messages[-1]["content"]
    if "deep verification" in c: return "verification: uncertain"
    return "concerning dual-use\nsafety: unsafe because it enables harm"

async def _run(router):
    cfg = load_config("config.yaml")
    mem = ContextMemory(research_plan=ResearchPlan(goal="g"))
    h = Hypothesis(id="G1", text="some hypothesis", title="T", source_strategy="s")
    mem.add_hypothesis(h)
    res = await ReflectionAgent().execute(
        Task(agent=AgentName.REFLECTION, action=TaskType.REVIEW_HYPOTHESIS, target_id="G1"),
        mem, FakeLLM(router), cfg)
    return mem, res

@pytest.mark.asyncio
async def test_reflection_safe_path():
    mem, res = await _run(_safe_router)
    assert len(res.reviews) == 2 and len(mem.reviews["G1"]) == 2
    assert mem.get("G1").safety == Safety.SAFE
    assert res.follow_ups[0].action == TaskType.ADD_TO_TOURNAMENT

@pytest.mark.asyncio
async def test_reflection_unsafe_sets_quarantine_flag():
    mem, res = await _run(_unsafe_router)
    assert mem.get("G1").safety == Safety.UNSAFE
    assert any(r.safety == Safety.UNSAFE for r in res.reviews)
    # agent still emits the follow-up; the Supervisor is what drops it
    assert res.follow_ups[0].action == TaskType.ADD_TO_TOURNAMENT
