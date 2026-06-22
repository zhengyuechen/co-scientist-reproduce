import pytest
from cosci.models import Task, AgentName, TaskType
from cosci.tasks import GlobalTaskQueue

def _t(priority, target):
    return Task(agent=AgentName.RANKING, action=TaskType.RUN_TOURNAMENT_BATCH,
                target_id=target, priority=priority)

@pytest.mark.asyncio
async def test_priority_then_fifo():
    q = GlobalTaskQueue()
    await q.put(_t(5, "a"))
    await q.put(_t(1, "b"))   # higher priority (lower number)
    await q.put(_t(5, "c"))
    assert (await q.get()).target_id == "b"  # priority 1 first
    assert (await q.get()).target_id == "a"  # then FIFO among priority 5
    assert (await q.get()).target_id == "c"
    assert q.empty() is True
