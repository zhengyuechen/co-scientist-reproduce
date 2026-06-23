import pytest
from cosci.agents.reflection import ReflectionAgent, _parse_novelty
from cosci.agents.meta_review import MetaReviewAgent, passes_novelty_gate
from cosci.memory import ContextMemory
from cosci.models import Hypothesis, ResearchPlan, Task, AgentName, TaskType
from cosci.config import load_config
from tests.fake_llm import FakeLLM


def _h(hid, novelty=None, verification=None, elo=1200.0):
    return Hypothesis(id=hid, text=f"text {hid}", title=hid, source_strategy="s",
                      novelty=novelty, verification=verification, elo_rating=elo)


def test_parse_novelty_extracts_and_clamps():
    assert _parse_novelty("Analysis...\nnovelty: 2\nsafety: safe") == 2.0
    assert _parse_novelty("novelty: 12") == 10.0    # clamp high
    assert _parse_novelty("novelty: 0") == 1.0      # clamp low
    assert _parse_novelty("no verdict here") is None


@pytest.mark.asyncio
async def test_reflection_parses_verdicts_into_scores_not_empty():
    cfg = load_config("config.yaml")
    mem = ContextMemory(research_plan=ResearchPlan(goal="g"))
    mem.add_hypothesis(Hypothesis(id="G1", text="a restatement of Diosi-Penrose", title="T", source_strategy="s"))

    def router(a, m):
        c = m[-1]["content"].lower()
        if "deep verification" in c:
            return "Assumption 1 is shaky.\nverification: invalidated"
        return "Closest model: Diosi-Penrose; equivalent under relabeling.\nnovelty: 2\nsafety: safe"

    await ReflectionAgent().execute(
        Task(agent=AgentName.REFLECTION, action=TaskType.REVIEW_HYPOTHESIS, target_id="G1"),
        mem, FakeLLM(router), cfg)

    h = mem.get("G1")
    assert h.novelty == 2.0 and h.verification == "invalidated"
    full = [r for r in mem.reviews["G1"] if r.type == "full"][0]
    deep = [r for r in mem.reviews["G1"] if r.type == "deep_verification"][0]
    assert full.scores == {"novelty": 2.0}          # the previously-empty scores object now carries the verdict
    assert deep.scores == {"verification": 0.0}


def test_passes_novelty_gate():
    assert passes_novelty_gate(_h("a", novelty=8), 5.0) is True
    assert passes_novelty_gate(_h("b", novelty=2), 5.0) is False              # restatement
    assert passes_novelty_gate(_h("c", verification="invalidated"), 5.0) is False
    assert passes_novelty_gate(_h("d", novelty=None), 5.0) is True            # fail open on missing score
    assert passes_novelty_gate(_h("e", novelty=5), 5.0) is True               # at threshold passes


@pytest.mark.asyncio
async def test_overview_excludes_low_novelty_even_if_it_won_the_tournament():
    cfg = load_config("config.yaml")                 # overview.min_novelty default 5.0
    mem = ContextMemory(research_plan=ResearchPlan(goal="is collapse physical"))
    mem.add_hypothesis(_h("G1", novelty=2, elo=1248))   # restatement, HIGHEST Elo (the G1-wins case)
    mem.add_hypothesis(_h("G2", novelty=8, elo=1200))   # genuinely novel, lower Elo
    captured = {}

    def router(a, m):
        captured["prompt"] = m[-1]["content"]
        return "Overview."

    await MetaReviewAgent().execute(
        Task(agent=AgentName.META_REVIEW, action=TaskType.GENERATE_FINAL_OVERVIEW),
        mem, FakeLLM(router), cfg)

    p = captured["prompt"]
    assert "text G2" in p                # the novel hypothesis reaches synthesis
    assert "text G1" not in p            # the restatement is excluded though it topped the tournament
    assert "excluded as low-novelty" in p


@pytest.mark.asyncio
async def test_overview_reports_when_nothing_clears_the_bar():
    cfg = load_config("config.yaml")
    mem = ContextMemory(research_plan=ResearchPlan(goal="g"))
    mem.add_hypothesis(_h("G1", novelty=2))
    mem.add_hypothesis(_h("G2", verification="invalidated", novelty=7))
    captured = {}

    def router(a, m):
        captured["prompt"] = m[-1]["content"]
        return "Overview."

    await MetaReviewAgent().execute(
        Task(agent=AgentName.META_REVIEW, action=TaskType.GENERATE_FINAL_OVERVIEW),
        mem, FakeLLM(router), cfg)

    assert "No candidate cleared the novelty bar" in captured["prompt"]
