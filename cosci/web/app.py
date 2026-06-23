"""FastAPI app: edit config, launch engine runs, and view results in the browser.

The web layer is fully separate from the engine — it only orchestrates the existing
package (load_config, run_engine, write_results). The run-execution core lives in
`execute_run`, which is awaitable directly so it can be tested without a live server.
"""
from __future__ import annotations

import asyncio
import csv
import glob
import json
from datetime import datetime
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from cosci.agents import (
    EvolutionAgent, GenerationAgent, MetaReviewAgent,
    ProximityAgent, RankingAgent, ReflectionAgent,
)
from cosci.config import Config, load_config
from cosci.engine import run_engine
from cosci.logging_utils import make_run_dir, write_results, slugify
from cosci.memory import ContextMemory
from cosci.models import AgentName
from cosci.tools.web_search import build_backend, is_faithful_grounding

STATIC = Path(__file__).parent / "static"
SNAP_BASE = "snapshots"


def _build_agents(grounding) -> dict:
    return {
        AgentName.GENERATION: GenerationAgent(grounding=grounding),
        AgentName.REFLECTION: ReflectionAgent(grounding=grounding),
        AgentName.RANKING: RankingAgent(),
        AgentName.EVOLUTION: EvolutionAgent(),
        AgentName.PROXIMITY: ProximityAgent(),
        AgentName.META_REVIEW: MetaReviewAgent(),
    }


def default_llm_factory(cfg: Config):
    from cosci.llm import OpenRouterClient
    return OpenRouterClient(cfg)


async def execute_run(run_id, goal, mode, cfg, llm, grounding, registry, results_base, timestamp):
    """Run the engine to completion and record results. Awaitable for tests."""
    rec = registry[run_id]
    rec["status"] = "running"
    try:
        Path(SNAP_BASE).mkdir(parents=True, exist_ok=True)
        mem = ContextMemory()
        agents = _build_agents(grounding)
        overview = await run_engine(goal, mem, llm, cfg, agents, mode=mode, snapshot_path=rec["snapshot"])
        if overview is None:
            rec["status"] = "aborted"
        else:
            out = make_run_dir(results_base, goal, timestamp)
            write_results(mem, overview, out)
            rec.update(status="done", out=out)
    except Exception as exc:  # surface the failure to the UI rather than crash the task
        rec.update(status="error", error=str(exc))


def _read_run(results_base: str, run_id: str) -> dict:
    d = Path(results_base) / run_id
    if not d.is_dir():
        raise HTTPException(404, f"run '{run_id}' not found")

    def load_json(name, default):
        p = d / name
        return json.loads(p.read_text()) if p.exists() else default

    overview = load_json("research_overview.json", {})
    tournament = []
    tj = d / "tournament.jsonl"
    if tj.exists():
        tournament = [json.loads(line) for line in tj.read_text().splitlines() if line.strip()]
    trajectory = []
    tc = d / "elo_trajectory.csv"
    if tc.exists():
        for row in csv.DictReader(tc.read_text().splitlines()):
            trajectory.append({"match": int(row["match"]), "tick": int(row["tick"]),
                               "best_elo": float(row["best_elo"])})
    return {
        "id": run_id,
        "goal": overview.get("goal"),
        "overview": overview.get("overview"),
        "hypotheses": load_json("hypotheses.json", []),
        "reviews": load_json("reviews.json", {}),
        "tournament": tournament,
        "elo_trajectory": trajectory,
    }


def create_app(llm_factory=default_llm_factory, config_path="config.yaml", results_base="results") -> FastAPI:
    app = FastAPI(title="Co-Scientist")
    registry: dict[str, dict] = {}
    app.state.registry = registry

    if STATIC.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text()

    @app.get("/api/config")
    def get_config():
        with open(config_path) as f:
            return {"config": yaml.safe_load(f), "faithful_grounding": _faithful(config_path)}

    @app.put("/api/config")
    async def put_config(request: Request):
        data = await request.json()
        try:
            Config(**data)
        except Exception as exc:
            raise HTTPException(400, f"invalid config: {exc}")
        with open(config_path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        return {"ok": True}

    @app.get("/api/runs")
    def list_runs():
        out = []
        for d in sorted(glob.glob(f"{results_base}/*/"), reverse=True):
            p = Path(d)
            ov = p / "research_overview.json"
            if not ov.exists():
                continue
            meta = json.loads(ov.read_text())
            hyps = []
            hp = p / "hypotheses.json"
            if hp.exists():
                hyps = json.loads(hp.read_text())
            elos = [h.get("elo_rating") for h in hyps if h.get("elo_rating") is not None]
            out.append({
                "id": p.name,
                "goal": meta.get("goal"),
                "n_hypotheses": len(hyps),
                "best_elo": max(elos) if elos else None,
            })
        return out

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        return _read_run(results_base, run_id)

    @app.post("/api/runs")
    async def launch_run(request: Request):
        body = await request.json()
        goal = (body.get("goal") or "").strip()
        if not goal:
            raise HTTPException(400, "a research goal is required")
        mode = body.get("mode", "continuous")
        cfg = load_config(config_path)
        try:
            llm = llm_factory(cfg)
        except Exception as exc:
            raise HTTPException(400, str(exc))
        grounding = build_backend(cfg)
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        run_id = f"{ts}_{slugify(goal)}"
        registry[run_id] = {"status": "queued", "goal": goal, "error": None,
                            "snapshot": str(Path(SNAP_BASE) / f"{run_id}.json")}
        asyncio.create_task(
            execute_run(run_id, goal, mode, cfg, llm, grounding, registry, results_base, ts)
        )
        return {"run_id": run_id}

    @app.get("/api/runs/{run_id}/status")
    def run_status(run_id: str):
        rec = registry.get(run_id)
        if rec is None:
            if (Path(results_base) / run_id).is_dir():
                return {"status": "done"}
            raise HTTPException(404, f"run '{run_id}' not found")
        live = {}
        snap = Path(rec["snapshot"])
        if snap.exists():
            try:
                mem = ContextMemory.load_snapshot(str(snap))
                active = mem.active_hypotheses()
                elos = [h.elo_rating for h in active if h.elo_rating is not None]
                live = {"n_hypotheses": len(active), "n_matches": len(mem.tournament),
                        "best_elo": max(elos) if elos else None}
            except Exception:
                pass
        return {"status": rec["status"], "error": rec.get("error"), **live}

    def _faithful(path):
        try:
            return is_faithful_grounding(load_config(path))
        except Exception:
            return False

    return app


app = create_app()
