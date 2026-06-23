import pytest
from cosci.engine import run_engine
from cosci.memory import ContextMemory
from cosci.config import load_config
from cosci.models import AgentName
from cosci.agents import (GenerationAgent, ReflectionAgent, RankingAgent,
                          EvolutionAgent, ProximityAgent, MetaReviewAgent)
from tests.fake_llm import FakeLLM

class _Enc:  # proximity fake encoder
    def encode(self, texts):
        return [[float(len(t)), 1.0] for t in texts]

def _agents():
    return {AgentName.GENERATION: GenerationAgent(strategies=["literature_review"]),
            AgentName.REFLECTION: ReflectionAgent(),
            AgentName.RANKING: RankingAgent(),
            AgentName.EVOLUTION: EvolutionAgent(),
            AgentName.PROXIMITY: ProximityAgent(encoder=_Enc()),
            AgentName.META_REVIEW: MetaReviewAgent()}

def _router(agent, messages):
    c = messages[-1]["content"].lower()
    if "safety" in c and "research goal" in c: return "safety: safe"
    if "structured research plan" in c or "preferences, attributes" in c:
        return '{"goal": "cure X", "preferences": ["novel"], "attributes": [], "constraints": []}'
    if "deep verification" in c: return "verification: verified"
    if "better idea" in c or "comparative analysis" in c or "better hypothesis" in c:
        return "debate\nbetter idea: 1"
    if "meta-analysis" in c or "research overview" in c or "research directions" in c:
        return "Overview: direction A."
    if "safety:" in c or "rapid initial" in c or "literature-grounded review" in c:
        return "looks fine\nsafety: safe"
    return "Hypothesis: mechanism M explains the effect in detail."

@pytest.mark.asyncio
async def test_engine_runs_to_completion_and_returns_overview(tmp_path):
    cfg = load_config("config.yaml")
    cfg.budget.max_ideas = 2; cfg.budget.max_matches_per_idea = 1   # tiny budget
    mem = ContextMemory()
    overview = await run_engine("cure X", mem, FakeLLM(_router), cfg, _agents(), mode="continuous")
    assert isinstance(overview, str) and len(overview) > 0
    assert len(mem.active_hypotheses()) >= 1          # generated hypotheses exist
    assert any(r for rs in mem.reviews.values() for r in rs)  # reviews happened
    # (a tournament match may or may not occur depending on timing; overview is the key contract)

@pytest.mark.asyncio
async def test_engine_aborts_on_unsafe_goal():
    cfg = load_config("config.yaml"); mem = ContextMemory()
    llm = FakeLLM(lambda a, m: "dangerous\nsafety: unsafe")
    assert await run_engine("bioweapon", mem, llm, cfg, _agents()) is None
