import os, pytest
from cosci.cli import run_cli, parse_args
from cosci.config import load_config
from cosci.models import AgentName
from tests.fake_llm import FakeLLM

def _router(agent, messages):
    c = messages[-1]["content"].lower()
    if "safety" in c and "research goal" in c: return "safety: safe"
    if "preferences, attributes" in c: return '{"goal":"g","preferences":[],"attributes":[],"constraints":[]}'
    if "deep verification" in c: return "verification: verified"
    if "better idea" in c or "comparative" in c: return "better idea: 1"
    if "research overview" in c or "research directions" in c or "meta-analysis" in c: return "Overview text."
    if "safety:" in c or "review" in c: return "safety: safe"
    return "Hypothesis: mechanism."

def test_parse_args_defaults():
    a = parse_args(["cure X"])
    assert a.goal == "cure X" and a.mode == "continuous"

@pytest.mark.asyncio
async def test_run_cli_writes_results(tmp_path):
    cfg = load_config("config.yaml"); cfg.budget.max_ideas = 2; cfg.budget.max_matches_per_idea = 1
    out = await run_cli("cure X", cfg, FakeLLM(_router), grounding=None,
                        results_base=str(tmp_path), timestamp="2026-06-22_120000")
    assert os.path.isfile(os.path.join(out, "research_overview.md"))
    assert os.path.isfile(os.path.join(out, "elo_trajectory.csv"))
    assert "Overview text." in open(os.path.join(out, "research_overview.md")).read()
