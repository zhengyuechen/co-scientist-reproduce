from cosci.memory import ContextMemory
from cosci.models import Hypothesis, Safety, ResearchPlan

def _h(hid, safety=Safety.SAFE, active=True):
    return Hypothesis(id=hid, text="t", title="T", source_strategy="s",
                      safety=safety, active=active)

def test_active_excludes_unsafe_and_inactive():
    m = ContextMemory(research_plan=ResearchPlan(goal="g"))
    m.add_hypothesis(_h("G1"))
    m.add_hypothesis(_h("G2", safety=Safety.UNSAFE))     # quarantined
    m.add_hypothesis(_h("G3", active=False))             # inactive
    ids = {h.id for h in m.active_hypotheses()}
    assert ids == {"G1"}

def test_snapshot_roundtrip(tmp_path):
    m = ContextMemory(research_plan=ResearchPlan(goal="g"))
    m.add_hypothesis(_h("G1"))
    p = tmp_path / "snap.json"
    m.save_snapshot(str(p))
    m2 = ContextMemory.load_snapshot(str(p))
    assert m2.get("G1").title == "T"
    assert m2.research_plan.goal == "g"

def test_new_id_is_unique():
    m = ContextMemory()
    a, b = m.new_id("G"), m.new_id("G")
    assert a != b and a.startswith("G")
