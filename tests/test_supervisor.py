import pytest
from cosci.supervisor import Supervisor
from cosci.memory import ContextMemory
from cosci.models import (Hypothesis, ResearchPlan, Task, AgentName, TaskType, Safety)
from cosci.agents.base import Results
from cosci.config import load_config
from tests.fake_llm import FakeLLM

def _sup():
    return Supervisor(agents={})  # route not exercised in these unit tests

@pytest.mark.asyncio
async def test_start_unsafe_goal_aborts():
    cfg = load_config("config.yaml"); mem = ContextMemory()
    llm = FakeLLM(lambda a, m: "this is dangerous\nsafety: unsafe")
    assert await _sup().start("make a bioweapon", mem, llm, cfg) is None

@pytest.mark.asyncio
async def test_start_safe_goal_parses_plan_and_seeds_generation():
    cfg = load_config("config.yaml"); mem = ContextMemory()
    def router(a, m):
        c = m[-1]["content"]
        if "safety" in c.lower() and "research goal" in c.lower(): return "safety: safe"
        return '{"goal": "cure X", "preferences": ["novel"], "attributes": [], "constraints": ["ethical"]}'
    task = await _sup().start("cure X", mem, FakeLLM(router), cfg)
    assert task.agent == AgentName.GENERATION and task.action == TaskType.CREATE_INITIAL_HYPOTHESES
    assert mem.research_plan.goal == "cure X" and mem.research_plan.preferences == ["novel"]

def test_manage_follow_ups_quarantines_unsafe():
    mem = ContextMemory()
    safe = Hypothesis(id="G1", text="t", title="T", source_strategy="s", safety=Safety.SAFE)
    unsafe = Hypothesis(id="G2", text="t", title="T", source_strategy="s", safety=Safety.UNSAFE)
    mem.add_hypothesis(safe); mem.add_hypothesis(unsafe)
    res = Results(follow_ups=[
        Task(agent=AgentName.RANKING, action=TaskType.ADD_TO_TOURNAMENT, target_id="G1"),
        Task(agent=AgentName.RANKING, action=TaskType.ADD_TO_TOURNAMENT, target_id="G2"),
    ])
    kept = _sup().manage_follow_ups(res, mem)
    assert [t.target_id for t in kept] == ["G1"]   # G2 (unsafe) dropped

def test_is_terminal_on_max_ideas():
    cfg = load_config("config.yaml"); mem = ContextMemory()
    for i in range(cfg.budget.max_ideas):
        mem.add_hypothesis(Hypothesis(id=f"G{i}", text="t", title="T", source_strategy="s"))
    assert _sup().is_terminal(mem, cfg) is True

def test_decide_next_steps_emits_tournament_when_two_in_tournament():
    cfg = load_config("config.yaml"); mem = ContextMemory()
    for hid in ("G1", "G2"):
        mem.add_hypothesis(Hypothesis(id=hid, text="t", title="T", source_strategy="s", elo_rating=1200))
    tasks = _sup().decide_next_steps(mem, cfg)
    assert any(t.action == TaskType.RUN_TOURNAMENT_BATCH for t in tasks)
