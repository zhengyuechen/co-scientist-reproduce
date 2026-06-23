"""Supervisor: SN8 task-chaining, safety quarantine, idle policy, termination."""
from __future__ import annotations
from cosci.agents.base import Agent, Results, parse_label
from cosci.prompts.reconstructed import SUP_SAFETY_REVIEW_GOAL, SUP_GOAL_PARSE
from cosci.prompts.render import render
from cosci.llm import LLMClient, extract_json
from cosci.models import (
    ResearchPlan, Task, AgentName, TaskType, Safety,
)
from cosci.memory import ContextMemory
from cosci.config import Config


class Supervisor:
    def __init__(self, agents: dict[AgentName, Agent]) -> None:
        self.agents = agents
        self._elo_hist: list[float] = []

    async def start(self, goal_raw: str, memory: ContextMemory, llm: LLMClient, cfg: Config) -> Task | None:
        # 1. Safety gate
        safety_prompt = render(SUP_SAFETY_REVIEW_GOAL, goal_raw=goal_raw)
        safety_resp = await llm.complete("meta_review", [{"role": "user", "content": safety_prompt}])
        if parse_label(safety_resp, "safety") == "unsafe":
            return None

        # 2. Parse goal into ResearchPlan
        parse_prompt = render(SUP_GOAL_PARSE, goal_raw=goal_raw)
        parse_resp = await llm.complete("meta_review", [{"role": "user", "content": parse_prompt}])
        try:
            data = extract_json(parse_resp)
        except ValueError:
            data = {}
        plan = ResearchPlan(
            goal=data.get("goal", goal_raw),
            preferences=data.get("preferences", []),
            attributes=data.get("attributes", []),
            constraints=data.get("constraints", []),
        )
        memory.research_plan = plan

        # 3. Seed generation
        return Task(agent=AgentName.GENERATION, action=TaskType.CREATE_INITIAL_HYPOTHESES)

    def route(self, task: Task) -> Agent:
        return self.agents[task.agent]

    def manage_follow_ups(self, results: Results, memory: ContextMemory) -> list[Task]:
        kept = []
        for ft in results.follow_ups:
            if ft.action == TaskType.ADD_TO_TOURNAMENT:
                h = memory.get(ft.target_id)
                if h is not None and h.safety == Safety.UNSAFE:
                    continue  # quarantine
            kept.append(ft)
        return kept

    def summary_stats(self, memory: ContextMemory) -> dict:
        active = memory.active_hypotheses()
        in_tourn = [h for h in active if h.elo_rating is not None]
        return {
            "num_ideas": len(active),
            "num_in_tournament": len(in_tourn),
            "matches_total": len(memory.tournament),
            "min_matches_per_idea": min((h.matches_played for h in in_tourn), default=0),
            "best_elo": max((h.elo_rating for h in in_tourn), default=None),
        }

    def is_terminal(self, memory: ContextMemory, cfg: Config) -> bool:
        stats = self.summary_stats(memory)
        if stats["num_ideas"] >= cfg.budget.max_ideas:
            return True
        in_tourn = stats["num_in_tournament"]
        if in_tourn > 0 and stats["min_matches_per_idea"] >= cfg.budget.max_matches_per_idea:
            return True
        if cfg.budget.max_tokens and memory.tokens_spent >= cfg.budget.max_tokens:
            return True
        return False

    def decide_next_steps(self, memory: ContextMemory, cfg: Config) -> list[Task]:
        stats = self.summary_stats(memory)
        best_elo = stats["best_elo"]
        if best_elo is not None:
            self._elo_hist.append(best_elo)

        active = memory.active_hypotheses()
        in_tourn = [h for h in active if h.elo_rating is not None]
        tasks: list[Task] = []

        if len(in_tourn) >= 2:
            tasks.append(Task(agent=AgentName.RANKING, action=TaskType.RUN_TOURNAMENT_BATCH))

        plateaued = (
            len(self._elo_hist) >= cfg.plateau.window
            and (self._elo_hist[-1] - self._elo_hist[-cfg.plateau.window]) <= cfg.plateau.epsilon
        )
        if plateaued and len(in_tourn) >= 2:
            tasks.append(Task(agent=AgentName.EVOLUTION, action=TaskType.EVOLVE_TOP))

        if len(in_tourn) >= 2 and memory.tick % 5 == 0:
            tasks.append(Task(agent=AgentName.PROXIMITY, action=TaskType.UPDATE_PROXIMITY))
            tasks.append(Task(agent=AgentName.META_REVIEW, action=TaskType.GENERATE_SYSTEM_FEEDBACK))

        return tasks
