"""Pure functions to write a run's results to a directory."""
from __future__ import annotations
import csv
import json
import os
import re


def slugify(text: str, maxlen: int = 40) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text[:maxlen]


def make_run_dir(base: str, goal: str, timestamp: str) -> str:
    path = f"{base}/{timestamp}_{slugify(goal)}"
    os.makedirs(path, exist_ok=True)
    return path


def grounding_stats(memory) -> dict:
    """How much of this run was grounded in retrieved literature.

    A review is grounded when the reflection agent had article sources to cite
    (``tool_grounded``); a 0/N here means arXiv returned nothing or rate-limited
    and the run fell back to parametric reasoning — the run is not literature-backed.
    """
    reviews = [r for rs in memory.reviews.values() for r in rs]
    hyp_grounded = sum(
        1 for rs in memory.reviews.values() if any(getattr(r, "tool_grounded", False) for r in rs)
    )
    return {
        "reviews_total": len(reviews),
        "reviews_grounded": sum(1 for r in reviews if getattr(r, "tool_grounded", False)),
        "hypotheses_total": len(memory.hypotheses),
        "hypotheses_grounded": hyp_grounded,
    }


def grounding_line(memory) -> str:
    s = grounding_stats(memory)
    return f"grounding: {s['reviews_grounded']}/{s['reviews_total']} reviews grounded"


def elo_trajectory(memory) -> list[dict]:
    if not memory.tournament:
        return []
    sorted_matches = sorted(memory.tournament, key=lambda m: m.tick)
    latest: dict[str, float] = {}
    result = []
    for i, m in enumerate(sorted_matches):
        latest.update(m.elo_after)
        result.append({"match": i + 1, "tick": m.tick, "best_elo": max(latest.values())})
    return result


def write_results(memory, overview: str, out_dir: str) -> None:
    # hypotheses.json — sorted by elo desc, None last
    hyps = sorted(
        memory.hypotheses.values(),
        key=lambda h: (h.elo_rating is not None, h.elo_rating or 0),
        reverse=True,
    )
    with open(os.path.join(out_dir, "hypotheses.json"), "w") as f:
        json.dump([h.model_dump() for h in hyps], f, indent=2, default=str)

    # reviews.json
    with open(os.path.join(out_dir, "reviews.json"), "w") as f:
        json.dump(
            {hid: [r.model_dump() for r in reviews] for hid, reviews in memory.reviews.items()},
            f, indent=2, default=str,
        )

    # tournament.jsonl
    with open(os.path.join(out_dir, "tournament.jsonl"), "w") as f:
        for m in memory.tournament:
            f.write(json.dumps(m.model_dump(), default=str) + "\n")

    # elo_trajectory.csv
    with open(os.path.join(out_dir, "elo_trajectory.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["match", "tick", "best_elo"])
        writer.writeheader()
        writer.writerows(elo_trajectory(memory))

    # research_overview.md
    with open(os.path.join(out_dir, "research_overview.md"), "w") as f:
        f.write("# Research overview\n\n" + (overview or ""))

    # research_overview.json
    with open(os.path.join(out_dir, "research_overview.json"), "w") as f:
        json.dump(
            {
                "goal": memory.research_plan.goal if memory.research_plan else None,
                "overview": overview,
                "grounding": grounding_stats(memory),
            },
            f, indent=2, default=str,
        )


def summary_line(memory) -> str:
    active = memory.active_hypotheses()
    elos = [h.elo_rating for h in active if h.elo_rating is not None]
    best = f"{max(elos):.1f}" if elos else "n/a"
    return f"{len(active)} hypotheses, {len(memory.tournament)} matches, best Elo {best}"
