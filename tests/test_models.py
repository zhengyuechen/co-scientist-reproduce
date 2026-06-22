from cosci.models import Hypothesis, Safety, Task, AgentName, TaskType

def test_hypothesis_defaults():
    h = Hypothesis(id="G1", text="t", title="T", source_strategy="literature_review")
    assert h.elo_rating is None
    assert h.active is True
    assert h.safety == Safety.UNREVIEWED
    assert h.origin == "generated"

def test_task_roundtrip_json():
    t = Task(agent=AgentName.REFLECTION, action=TaskType.REVIEW_HYPOTHESIS, target_id="G1")
    dumped = t.model_dump_json()
    assert '"target_id":"G1"' in dumped.replace(" ", "")
