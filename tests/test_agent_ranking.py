import pytest
from cosci.agents.ranking import RankingAgent
from cosci.elo import INITIAL_ELO
from cosci.memory import ContextMemory
from cosci.models import Hypothesis, ResearchPlan, Task, AgentName, TaskType, DebateMode
from cosci.config import load_config
from tests.fake_llm import FakeLLM

def _mk(mem, hid):
    h = Hypothesis(id=hid, text=f"text {hid}", title=hid, source_strategy="s")
    mem.add_hypothesis(h); return h

@pytest.mark.asyncio
async def test_add_to_tournament_sets_initial_elo_idempotent():
    cfg = load_config("config.yaml"); mem = ContextMemory()
    _mk(mem, "G1")
    agent = RankingAgent()
    await agent.execute(Task(agent=AgentName.RANKING, action=TaskType.ADD_TO_TOURNAMENT, target_id="G1"),
                        mem, FakeLLM(lambda a, m: ""), cfg)
    assert mem.get("G1").elo_rating == INITIAL_ELO
    mem.get("G1").elo_rating = 1400  # idempotent: second add must not reset
    await agent.execute(Task(agent=AgentName.RANKING, action=TaskType.ADD_TO_TOURNAMENT, target_id="G1"),
                        mem, FakeLLM(lambda a, m: ""), cfg)
    assert mem.get("G1").elo_rating == 1400

@pytest.mark.asyncio
async def test_run_batch_updates_elo_and_records_match():
    cfg = load_config("config.yaml")
    mem = ContextMemory(research_plan=ResearchPlan(goal="g"))
    for hid in ("G1", "G2"):
        _mk(mem, hid); mem.get(hid).elo_rating = INITIAL_ELO
    res = await RankingAgent().execute(
        Task(agent=AgentName.RANKING, action=TaskType.RUN_TOURNAMENT_BATCH),
        mem, FakeLLM(lambda a, m: "debate...\nbetter idea: 1"), cfg)
    assert res.match is not None
    win, lose = res.match.winner_id, res.match.loser_id
    assert mem.get(win).elo_rating > INITIAL_ELO and mem.get(lose).elo_rating < INITIAL_ELO
    assert mem.get(win).matches_played == 1 and len(mem.tournament) == 1
