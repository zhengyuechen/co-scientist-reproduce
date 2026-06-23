"""Engine: the queue-driven run loop that integrates the six agents + Supervisor.

continuous mode (primary): a single sequential worker pops tasks off a global
priority queue, routes each to its agent, advances the tick, and enqueues the
agent's follow-ups (after the Supervisor's safety quarantine). When the queue
drains, the Supervisor decides the next steps; if it has nothing to do for
several rounds, or a budget cap trips ``is_terminal``, the loop ends. Faithful
to SN8's StartCoScientist: sequential, not concurrent (the lock is a
correctness belt-and-suspenders for a future concurrent dispatcher).

round_based mode: a simpler deterministic phase loop (generate once, drain
reviews, add to tournament, run a tournament batch, and periodically evolve +
update proximity), bounded by ``max_ideas`` rounds and ``is_terminal``.

Both modes finish by asking the meta-review agent for the final research
overview and returning its text.
"""
from __future__ import annotations
import asyncio
import logging

from cosci.supervisor import Supervisor
from cosci.tasks import GlobalTaskQueue
from cosci.models import Task, AgentName, TaskType
from cosci.memory import ContextMemory
from cosci.llm import LLMClient
from cosci.config import Config
from cosci.agents.base import Agent

log = logging.getLogger(__name__)


async def _final_overview(agents: dict[AgentName, Agent], memory: ContextMemory,
                          llm: LLMClient, cfg: Config) -> str | None:
    task = Task(agent=AgentName.META_REVIEW, action=TaskType.GENERATE_FINAL_OVERVIEW)
    ov = await agents[AgentName.META_REVIEW].execute(task, memory, llm, cfg)
    return ov.overview


async def _run_continuous(sup: Supervisor, lock: asyncio.Lock, initial: Task,
                          memory: ContextMemory, llm: LLMClient, cfg: Config,
                          snapshot_path: str | None) -> None:
    queue = GlobalTaskQueue()
    await queue.put(initial)
    idle = 0
    max_iters = 1000 + 50 * cfg.budget.max_ideas * (cfg.budget.max_matches_per_idea + 1)
    iters = 0
    while not sup.is_terminal(memory, cfg):
        iters += 1
        if iters > max_iters:
            log.warning("engine: hit hard iteration cap (%d); terminating", max_iters)
            break
        if queue.empty():
            nxt = sup.decide_next_steps(memory, cfg)
            if not nxt:
                idle += 1
                if idle > 3:
                    break
                continue
            idle = 0
            for t in nxt:
                await queue.put(t)
            continue
        t = await queue.get()
        async with lock:
            agent = sup.route(t)
            res = await agent.execute(t, memory, llm, cfg)
            memory.tick += 1
            for ft in sup.manage_follow_ups(res, memory):
                await queue.put(ft)
            if snapshot_path and memory.tick % 10 == 0:
                memory.save_snapshot(snapshot_path)


async def _run_one(sup: Supervisor, lock: asyncio.Lock, task: Task,
                   memory: ContextMemory, llm: LLMClient, cfg: Config) -> list[Task]:
    async with lock:
        agent = sup.route(task)
        res = await agent.execute(task, memory, llm, cfg)
        memory.tick += 1
        return sup.manage_follow_ups(res, memory)


async def _run_round_based(sup: Supervisor, lock: asyncio.Lock, initial: Task,
                           memory: ContextMemory, llm: LLMClient, cfg: Config) -> None:
    rnd = 0
    while rnd < cfg.budget.max_ideas and not sup.is_terminal(memory, cfg):
        # Generation: first round seeds, later rounds re-run the seed task.
        review_tasks = await _run_one(sup, lock, initial, memory, llm, cfg)

        # Drain the resulting REVIEW follow-ups; collect their ADD_TO_TOURNAMENT chains.
        tourn_tasks: list[Task] = []
        for rt in review_tasks:
            if rt.action == TaskType.REVIEW_HYPOTHESIS:
                tourn_tasks += await _run_one(sup, lock, rt, memory, llm, cfg)

        # Add each reviewed hypothesis to the tournament.
        for at in tourn_tasks:
            if at.action == TaskType.ADD_TO_TOURNAMENT:
                await _run_one(sup, lock, at, memory, llm, cfg)

        # Run a tournament batch.
        await _run_one(
            sup, lock,
            Task(agent=AgentName.RANKING, action=TaskType.RUN_TOURNAMENT_BATCH),
            memory, llm, cfg,
        )

        # Every other round: evolve the top hypotheses and refresh proximity.
        if rnd % 2 == 1:
            await _run_one(
                sup, lock,
                Task(agent=AgentName.EVOLUTION, action=TaskType.EVOLVE_TOP),
                memory, llm, cfg,
            )
            await _run_one(
                sup, lock,
                Task(agent=AgentName.PROXIMITY, action=TaskType.UPDATE_PROXIMITY),
                memory, llm, cfg,
            )

        rnd += 1


async def run_engine(goal_raw: str, memory: ContextMemory, llm: LLMClient, cfg: Config,
                     agents: dict[AgentName, Agent], mode: str = "continuous",
                     snapshot_path: str | None = None) -> str | None:
    sup = Supervisor(agents)
    initial = await sup.start(goal_raw, memory, llm, cfg)
    if initial is None:
        return None  # unsafe-goal abort: no overview

    lock = asyncio.Lock()
    if mode == "round_based":
        await _run_round_based(sup, lock, initial, memory, llm, cfg)
    else:
        await _run_continuous(sup, lock, initial, memory, llm, cfg, snapshot_path)

    return await _final_overview(agents, memory, llm, cfg)
