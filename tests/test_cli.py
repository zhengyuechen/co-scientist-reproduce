import os, pytest
from cosci import run_log
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
    out, _mem = await run_cli("cure X", cfg, FakeLLM(_router), grounding=None,
                              results_base=str(tmp_path), timestamp="2026-06-22_120000")
    assert os.path.isfile(os.path.join(out, "research_overview.md"))
    assert os.path.isfile(os.path.join(out, "elo_trajectory.csv"))
    assert "Overview text." in open(os.path.join(out, "research_overview.md")).read()
    # the run also emitted an event log bookended by run_started/run_done
    evs = run_log.read_events(run_log.events_path(str(tmp_path), "2026-06-22_120000_cure-x"))
    names = [e["event"] for e in evs]
    assert names[0] == "run_started" and names[-1] == "run_done"
    assert "generation_started" in names and "task" in names

@pytest.mark.asyncio
async def test_run_cli_unsafe_goal_writes_no_results(tmp_path):
    cfg = load_config("config.yaml")
    llm = FakeLLM(lambda a, m: "dangerous\nsafety: unsafe")  # goal safety-gate -> unsafe -> abort
    out, _mem = await run_cli("bioweapon", cfg, llm, grounding=None,
                              results_base=str(tmp_path), timestamp="2026-06-22_120000")
    assert out is None
    # no results directory is written on abort (only the event log records the attempt)
    assert not (tmp_path / "2026-06-22_120000_bioweapon").exists()
    evs = run_log.read_events(run_log.events_path(str(tmp_path), "2026-06-22_120000_bioweapon"))
    assert [e["event"] for e in evs] == ["run_started", "run_aborted"]
