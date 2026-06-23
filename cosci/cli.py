"""Command-line entrypoint for the Co-Scientist system."""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime

from cosci.agents import (
    GenerationAgent, ReflectionAgent, RankingAgent,
    EvolutionAgent, ProximityAgent, MetaReviewAgent,
)
from cosci import run_log
from cosci.config import Config, load_config
from cosci.engine import run_engine
from cosci.logging_utils import make_run_dir, write_results, summary_line, grounding_line, slugify
from cosci.memory import ContextMemory
from cosci.models import AgentName
from cosci.tools.web_search import build_backend, is_faithful_grounding


def build_agents(cfg: Config, grounding) -> dict:
    return {
        AgentName.GENERATION: GenerationAgent(grounding=grounding),
        AgentName.REFLECTION: ReflectionAgent(grounding=grounding),
        AgentName.RANKING: RankingAgent(),
        AgentName.EVOLUTION: EvolutionAgent(),
        AgentName.PROXIMITY: ProximityAgent(),
        AgentName.META_REVIEW: MetaReviewAgent(),
    }


async def run_cli(
    goal: str,
    cfg: Config,
    llm,
    *,
    grounding=None,
    mode: str = "continuous",
    results_base: str = "results",
    timestamp: str,
) -> tuple[str | None, ContextMemory]:
    run_log.bind(run_log.events_path(results_base, f"{timestamp}_{slugify(goal)}"))
    run_log.emit("run_started", goal=goal, mode=mode)
    mem = ContextMemory()
    agents = build_agents(cfg, grounding)
    overview = await run_engine(goal, mem, llm, cfg, agents, mode=mode)
    if overview is None:           # unsafe-goal abort -> no results written
        run_log.emit("run_aborted", reason="unsafe goal")
        return None, mem
    out = make_run_dir(results_base, goal, timestamp)
    write_results(mem, overview, out)
    run_log.emit("run_done", hypotheses=len(mem.hypotheses), matches=len(mem.tournament))
    return out, mem


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cosci.cli",
        description="Run the Co-Scientist multi-agent system on a research goal.",
    )
    parser.add_argument("goal", help="Research goal to investigate.")
    parser.add_argument(
        "--mode",
        choices=["continuous", "round_based"],
        default="continuous",
        help="Engine mode (default: continuous).",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config YAML (default: config.yaml).",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        dest="results_dir",
        help="Base directory for run output (default: results).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)

    if not is_faithful_grounding(cfg):
        backend = cfg.grounding.backend
        if backend == "none":
            print("Warning: grounding is disabled (parametric only) — no literature retrieval; "
                  "agents rely on the model's built-in knowledge. Fidelity is reduced.", file=sys.stderr)
        else:  # arxiv
            print("Warning: grounding is arXiv-only (not the broad web search used in the paper). "
                  "Fidelity is reduced. Set backend=tavily in config.yaml + WEB_SEARCH_API_KEY for "
                  "faithful grounding.", file=sys.stderr)

    from cosci.llm import OpenRouterClient
    llm = OpenRouterClient(cfg)
    grounding = build_backend(cfg)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    out, mem = asyncio.run(run_cli(args.goal, cfg, llm, grounding=grounding,
                                   mode=args.mode, results_base=args.results_dir, timestamp=ts))
    if out is None:
        print("Run aborted: the research goal did not pass the safety review.", file=sys.stderr)
        return 2
    print(f"Results written to {out}")
    print(summary_line(mem))
    print(grounding_line(mem))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
