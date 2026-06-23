import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cosci.web.app import create_app, execute_run
from cosci.config import load_config
from cosci.logging_utils import make_run_dir, write_results
from cosci.memory import ContextMemory
from cosci.models import Hypothesis, ResearchPlan
from tests.fake_llm import FakeLLM


def _router(agent, messages):
    c = messages[-1]["content"].lower()
    if "safety" in c and "research goal" in c:
        return "safety: safe"
    if "preferences, attributes" in c:
        return '{"goal":"g","preferences":[],"attributes":[],"constraints":[]}'
    if "deep verification" in c:
        return "verification: verified"
    if "better idea" in c or "comparative" in c:
        return "better idea: 1"
    if "research overview" in c or "research directions" in c or "meta-analysis" in c:
        return "Overview text."
    if "safety:" in c or "review" in c:
        return "safety: safe"
    return "Hypothesis: mechanism."


@pytest.fixture
def env(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    shutil.copy("config.yaml", cfg_path)
    results = tmp_path / "results"
    return cfg_path, results


def _client(env):
    cfg_path, results = env
    return TestClient(create_app(llm_factory=lambda cfg: FakeLLM(_router),
                                 config_path=str(cfg_path), results_base=str(results)))


def test_get_config(env):
    r = _client(env).get("/api/config")
    assert r.status_code == 200
    body = r.json()
    assert body["config"]["elo"]["k_factor"] == 32
    assert body["faithful_grounding"] is False  # arxiv default is not "faithful"


def test_put_config_valid_then_invalid(env):
    c = _client(env)
    cfg = c.get("/api/config").json()["config"]
    cfg["budget"]["max_ideas"] = 7
    assert c.put("/api/config", json=cfg).status_code == 200
    assert c.get("/api/config").json()["config"]["budget"]["max_ideas"] == 7
    assert c.put("/api/config", json={"elo": {"k_factor": 1}}).status_code == 400  # missing default_model


def test_runs_list_and_detail(env):
    _, results = env
    mem = ContextMemory(research_plan=ResearchPlan(goal="cure X"))
    mem.add_hypothesis(Hypothesis(id="G1", text="t", title="T1", source_strategy="s", elo_rating=1250.0))
    out = make_run_dir(str(results), "cure X", "2026-06-22_120000")
    write_results(mem, "Overview body.", out)
    c = _client(env)
    runs = c.get("/api/runs").json()
    assert len(runs) == 1 and runs[0]["goal"] == "cure X" and round(runs[0]["best_elo"]) == 1250
    detail = c.get(f"/api/runs/{runs[0]['id']}").json()
    assert detail["overview"] == "Overview body." and detail["hypotheses"][0]["id"] == "G1"
    # grounding status surfaces to the UI (this run has no reviews -> 0/0)
    assert detail["grounding"] == {"reviews_total": 0, "reviews_grounded": 0,
                                   "hypotheses_total": 1, "hypotheses_grounded": 0}


def test_launch_endpoint_returns_run_id(env):
    c = _client(env)
    cfg = c.get("/api/config").json()["config"]
    cfg["grounding"]["backend"] = "none"  # avoid any arXiv network from the background task
    c.put("/api/config", json=cfg)
    r = c.post("/api/runs", json={"goal": "cure X", "mode": "continuous"})
    assert r.status_code == 200 and r.json()["run_id"].endswith("cure-x")
    assert c.post("/api/runs", json={"goal": ""}).status_code == 400  # empty goal rejected
    assert c.post("/api/runs", json={"goal": "cure X", "mode": "bogus"}).status_code == 400  # bad mode rejected


def test_same_goal_launches_get_distinct_run_ids(env):
    c = _client(env)
    cfg = c.get("/api/config").json()["config"]
    cfg["grounding"]["backend"] = "none"
    c.put("/api/config", json=cfg)
    id1 = c.post("/api/runs", json={"goal": "cure X"}).json()["run_id"]
    id2 = c.post("/api/runs", json={"goal": "cure X"}).json()["run_id"]
    assert id1 != id2  # no collision even within the same second


@pytest.mark.asyncio
async def test_execute_run_writes_results(env):
    cfg_path, results = env
    cfg = load_config(str(cfg_path))
    cfg.budget.max_ideas = 2
    cfg.budget.max_matches_per_idea = 1
    Path(results).mkdir(parents=True, exist_ok=True)
    registry = {"r1": {"status": "queued", "snapshot": str(Path(results) / "r1.snap.json")}}
    # grounding=None -> parametric, no network
    await execute_run("r1", "cure X", "continuous", cfg, FakeLLM(_router), None,
                      registry, str(results), "2026-06-22_120000")
    assert registry["r1"]["status"] == "done"
    assert (Path(results) / "2026-06-22_120000_cure-x" / "research_overview.md").exists()


@pytest.mark.asyncio
async def test_execute_run_unsafe_goal_aborts(env):
    cfg_path, results = env
    cfg = load_config(str(cfg_path))
    Path(results).mkdir(parents=True, exist_ok=True)
    registry = {"r2": {"status": "queued", "snapshot": str(Path(results) / "r2.snap.json")}}
    await execute_run("r2", "bioweapon", "continuous", cfg, FakeLLM(lambda a, m: "safety: unsafe"),
                      None, registry, str(results), "2026-06-22_120000")
    assert registry["r2"]["status"] == "aborted"
    assert not (Path(results) / "2026-06-22_120000_bioweapon").exists()
