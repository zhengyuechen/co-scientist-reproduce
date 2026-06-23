"""Ranking agent: Elo tournament via LLM debate."""
from __future__ import annotations
import logging
import statistics

from cosci import run_log
from cosci.agents.base import Results, parse_label
from cosci.elo import INITIAL_ELO, update as elo_update
from cosci.memory import ContextMemory
from cosci.models import DebateMode, MatchResult, Task, TaskType
from cosci.prompts.verbatim import RANK_SINGLE_TURN, RANK_MULTI_TURN
from cosci.prompts.render import render

log = logging.getLogger(__name__)


class RankingAgent:
    async def execute(self, task: Task, memory: ContextMemory, llm, cfg) -> Results:
        if task.action == TaskType.ADD_TO_TOURNAMENT:
            h = memory.get(task.target_id)
            if h.elo_rating is None:
                h.elo_rating = INITIAL_ELO
            return Results()

        # RUN_TOURNAMENT_BATCH
        candidates = [h for h in memory.active_hypotheses() if h.elo_rating is not None]
        if len(candidates) < 2:
            return Results()

        # Selection: sort by (matches_played, -elo_rating), take first two
        candidates.sort(key=lambda h: (h.matches_played, -h.elo_rating))
        a, b = candidates[0], candidates[1]

        # Mode: multi_turn if BOTH are in top half by Elo
        elos = [h.elo_rating for h in candidates]
        median = statistics.median(elos)
        if a.elo_rating >= median and b.elo_rating >= median:
            mode = DebateMode.MULTI_TURN
            template = RANK_MULTI_TURN
        else:
            mode = DebateMode.SINGLE_TURN
            template = RANK_SINGLE_TURN

        plan = memory.research_plan
        goal = plan.goal if plan else ""
        preferences = "\n".join(plan.preferences) if plan else ""

        def latest_review(hid: str) -> str:
            texts = memory.reviews.get(hid, [])
            return texts[-1].text if texts else ""

        prompt = render(
            template,
            goal=goal,
            preferences=preferences,
            hypothesis_1=a.text,
            hypothesis_2=b.text,
            review_1=latest_review(a.id),
            review_2=latest_review(b.id),
            notes="",
            idea_attributes="",
        )

        resp = await llm.complete("ranking", [{"role": "user", "content": prompt}])

        # Parse winner
        w = parse_label(resp, "better idea", "better hypothesis")
        if w == "2":
            winner, loser = b, a
        else:
            if w != "1":
                log.warning("parse_label returned %r; defaulting to hypothesis_1", w)
            winner, loser = a, b

        elo_before = {a.id: a.elo_rating, b.id: b.elo_rating}
        new_w, new_l = elo_update(
            winner.elo_rating, loser.elo_rating,
            k=cfg.elo.k_factor, scale=cfg.elo.scale,
        )
        winner.elo_rating = new_w
        loser.elo_rating = new_l
        a.matches_played += 1
        b.matches_played += 1

        match = MatchResult(
            a_id=a.id,
            b_id=b.id,
            mode=mode,
            winner_id=winner.id,
            loser_id=loser.id,
            elo_before=elo_before,
            elo_after={winner.id: new_w, loser.id: new_l},
            transcript=resp,
            tick=memory.tick,
        )
        memory.record_match(match)
        run_log.emit("ranking_match", tick=memory.tick, a=a.id, b=b.id, winner=winner.id,
                     mode=mode, elo_delta=round(new_w - elo_before[winner.id], 1))
        return Results(match=match)
