import pytest
from cosci.agents.generation import GenerationAgent
from cosci.agents.base import Results
from cosci.memory import ContextMemory
from cosci.models import ResearchPlan, Task, AgentName, TaskType
from cosci.config import load_config
from tests.fake_llm import FakeLLM

def _router(agent, messages):
    # debate returns a HYPOTHESIS-marked response; others plain prose
    content = messages[-1]["content"]
    if "simulated discussion" in content or "collaborative discourse" in content:
        return "...debate...\nHYPOTHESIS\nNovel mechanism A explains the effect."
    return "Hypothesis: mechanism B drives the observation in detail."

@pytest.mark.asyncio
async def test_generation_makes_one_hypothesis_per_strategy(tmp_path, monkeypatch):
    cfg = load_config("config.yaml")
    mem = ContextMemory(research_plan=ResearchPlan(goal="cure X", preferences=["novel"]))
    agent = GenerationAgent(strategies=["literature_review", "scientific_debate"])
    task = Task(agent=AgentName.GENERATION, action=TaskType.CREATE_INITIAL_HYPOTHESES)
    res = await agent.execute(task, mem, FakeLLM(_router), cfg)
    assert len(res.new_hypotheses) == 2
    assert {h.source_strategy for h in res.new_hypotheses} == {"literature_review", "scientific_debate"}
    # debate hypothesis text taken after the HYPOTHESIS marker
    deb = [h for h in res.new_hypotheses if h.source_strategy == "scientific_debate"][0]
    assert deb.text.strip().startswith("Novel mechanism A")
    # each is in memory and each emits a reflection follow-up
    assert all(mem.get(h.id) is not None for h in res.new_hypotheses)
    assert len(res.follow_ups) == 2
    assert all(f.action == TaskType.REVIEW_HYPOTHESIS for f in res.follow_ups)


_BUNDLE = (
    "I'll initiate this discourse by proposing three distinct hypotheses.\n\n"
    "**Hypothesis 1: Thermodynamic boundary**\n\n"
    "Collapse occurs at an entropy threshold.\n\n"
    "**Hypothesis 2: Gravitational decoherence**\n\n"
    "Collapse follows a Diosi-Penrose timescale.\n\n"
    "**Hypothesis 3: Relational horizon**\n\n"
    "Collapse is enforced at the observer's perceptual state."
)


@pytest.mark.asyncio
async def test_generation_splits_bundled_proposals_into_atomic_hypotheses():
    cfg = load_config("config.yaml")
    mem = ContextMemory(research_plan=ResearchPlan(goal="is collapse physical", preferences=["novel"]))
    agent = GenerationAgent(strategies=["scientific_debate"])
    task = Task(agent=AgentName.GENERATION, action=TaskType.CREATE_INITIAL_HYPOTHESES)
    res = await agent.execute(task, mem, FakeLLM(lambda a, m: _BUNDLE), cfg)
    # one bundled response -> three first-class hypotheses, each with a follow-up review
    assert len(res.new_hypotheses) == 3
    assert len(res.follow_ups) == 3
    titles = [h.title for h in res.new_hypotheses]
    assert titles == ["Thermodynamic boundary", "Gravitational decoherence", "Relational horizon"]
    # no markdown or preamble leaked into a title
    assert all("*" not in t and "I'll initiate" not in t for t in titles)


# Debate that weighs several numbered candidates and THEN converges on one final
# HYPOTHESIS. Must yield the final idea, not the deliberation's candidate list.
_DELIBERATION_THEN_CONVERGED = (
    "Let me weigh several candidates before converging.\n\n"
    "**Hypothesis 1: Thermodynamic boundary**\n"
    "Strong on entropy but weak on locality.\n\n"
    "**Hypothesis 2: Gravitational decoherence**\n"
    "Elegant but hard to test.\n\n"
    "After debate, I synthesize the strongest elements.\n\n"
    "HYPOTHESIS\n"
    "Collapse is an entropy-threshold transition sharpened by gravitational self-energy."
)


@pytest.mark.asyncio
async def test_generation_debate_converges_past_deliberation_candidates():
    cfg = load_config("config.yaml")
    mem = ContextMemory(research_plan=ResearchPlan(goal="g", preferences=["novel"]))
    agent = GenerationAgent(strategies=["scientific_debate"])
    task = Task(agent=AgentName.GENERATION, action=TaskType.CREATE_INITIAL_HYPOTHESES)
    res = await agent.execute(task, mem, FakeLLM(lambda a, m: _DELIBERATION_THEN_CONVERGED), cfg)
    assert len(res.new_hypotheses) == 1                     # final idea, not the 2 candidates
    h = res.new_hypotheses[0]
    assert h.text.startswith("Collapse is an entropy-threshold")
    assert "Hypothesis 1" not in h.text and "Hypothesis 2" not in h.text


# Rare: the converged final answer is itself a numbered bundle -> split that.
_CONVERGED_BUNDLE = (
    "Discussion of the merits...\n\n"
    "HYPOTHESIS\n\n"
    "**Hypothesis 1: Alpha mechanism**\nDetails A.\n\n"
    "**Hypothesis 2: Beta mechanism**\nDetails B."
)


@pytest.mark.asyncio
async def test_generation_debate_splits_a_bundled_final_answer():
    cfg = load_config("config.yaml")
    mem = ContextMemory(research_plan=ResearchPlan(goal="g", preferences=["novel"]))
    agent = GenerationAgent(strategies=["scientific_debate"])
    task = Task(agent=AgentName.GENERATION, action=TaskType.CREATE_INITIAL_HYPOTHESES)
    res = await agent.execute(task, mem, FakeLLM(lambda a, m: _CONVERGED_BUNDLE), cfg)
    assert [h.title for h in res.new_hypotheses] == ["Alpha mechanism", "Beta mechanism"]
    assert all("Discussion of the merits" not in h.text for h in res.new_hypotheses)
