import pytest
from cosci.agents.proximity import ProximityAgent
from cosci.memory import ContextMemory
from cosci.models import Hypothesis, Task, AgentName, TaskType

def _mk(mem, hid, text):
    mem.add_hypothesis(Hypothesis(id=hid, text=text, title=hid, source_strategy="s"))

class FakeEncoder:
    """Maps known texts to fixed 2D unit vectors; no sentence-transformers needed."""
    def encode(self, texts):
        table = {"alpha": [1.0, 0.0], "beta": [0.0, 1.0]}
        return [table.get(t, [1.0, 0.0]) for t in texts]

@pytest.mark.asyncio
async def test_proximity_similarity_graph():
    mem = ContextMemory()
    _mk(mem, "G1", "alpha"); _mk(mem, "G2", "alpha"); _mk(mem, "G3", "beta")
    await ProximityAgent(encoder=FakeEncoder()).execute(
        Task(agent=AgentName.PROXIMITY, action=TaskType.UPDATE_PROXIMITY), mem, None, None)
    g1 = {e["other_id"]: e["similarity"] for e in mem.proximity["G1"]}
    assert abs(g1["G2"] - 1.0) < 1e-6   # identical text -> cosine 1
    assert abs(g1["G3"] - 0.0) < 1e-6   # orthogonal -> cosine 0

@pytest.mark.asyncio
async def test_proximity_noop_with_fewer_than_two():
    mem = ContextMemory()
    _mk(mem, "G1", "alpha")
    await ProximityAgent(encoder=FakeEncoder()).execute(
        Task(agent=AgentName.PROXIMITY, action=TaskType.UPDATE_PROXIMITY), mem, None, None)
    assert mem.proximity == {}
