import pytest
from cosci.engine import run_engine
from cosci.memory import ContextMemory
from cosci.config import load_config
from cosci.models import AgentName, Safety
from cosci.agents import (GenerationAgent, ReflectionAgent, RankingAgent,
                          EvolutionAgent, ProximityAgent, MetaReviewAgent)
from tests.fake_llm import FakeLLM

class _Enc:
    def encode(self, texts): return [[float(len(t)), 1.0] for t in texts]

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
    if "preferences, attributes" in c or "structured research plan" in c:
        return '{"goal": "g", "preferences": [], "attributes": [], "constraints": []}'
    if "deep verification" in c: return "verification: verified"
    if "better idea" in c or "comparative" in c: return "better idea: 1"
    if "meta-analysis" in c or "research overview" in c or "research directions" in c: return "Overview text."
    if "safety:" in c or "review" in c: return "safety: safe"
    return "Hypothesis: mechanism."

@pytest.mark.asyncio
async def test_round_based_reviews_evolved_hypotheses():
    cfg = load_config("config.yaml"); cfg.budget.max_ideas = 3; cfg.budget.max_matches_per_idea = 1
    mem = ContextMemory()
    ov = await run_engine("g", mem, FakeLLM(_router), cfg, _agents(), mode="round_based")
    assert isinstance(ov, str)
    evolved = [h for h in mem.hypotheses.values() if h.origin.value == "evolved"]
    # every evolved hypothesis got reviewed (safety set away from the unreviewed default)
    assert all(h.safety != Safety.UNREVIEWED for h in evolved)

@pytest.mark.asyncio
async def test_overview_empty_guard_not_none_on_completion():
    cfg = load_config("config.yaml"); cfg.budget.max_ideas = 1; cfg.budget.max_matches_per_idea = 1
    mem = ContextMemory()
    def router(a, m):
        c = m[-1]["content"].lower()
        if "safety" in c and "research goal" in c: return "safety: safe"
        if "preferences, attributes" in c: return '{"goal":"g","preferences":[],"attributes":[],"constraints":[]}'
        if "research overview" in c or "research directions" in c or "meta-analysis" in c: return ""  # empty overview
        if "deep verification" in c: return "verification: verified"
        if "safety:" in c or "review" in c: return "safety: safe"
        return "Hypothesis: x"
    ov = await run_engine("g", mem, FakeLLM(router), cfg, _agents(), mode="continuous")
    assert ov == ""           # completed run with empty overview → "" not None
