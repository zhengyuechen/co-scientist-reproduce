import json, os
from cosci.logging_utils import (slugify, make_run_dir, elo_trajectory, write_results,
                                 summary_line, grounding_stats, grounding_line)
from cosci.memory import ContextMemory
from cosci.models import Hypothesis, Review, MatchResult, ResearchPlan, DebateMode

def test_slugify():
    assert slugify("Cure X, fast!") == "cure-x-fast"

def test_make_run_dir(tmp_path):
    p = make_run_dir(str(tmp_path), "Cure X", "2026-06-22_120000")
    assert os.path.isdir(p) and p.endswith("2026-06-22_120000_cure-x")

def _mem():
    m = ContextMemory(research_plan=ResearchPlan(goal="g"))
    m.add_hypothesis(Hypothesis(id="G1", text="t1", title="T1", source_strategy="s", elo_rating=1216.0))
    m.add_hypothesis(Hypothesis(id="G2", text="t2", title="T2", source_strategy="s", elo_rating=1184.0))
    m.add_review(Review(hypothesis_id="G1", type="full", text="ok"))
    m.record_match(MatchResult(a_id="G1", b_id="G2", mode=DebateMode.SINGLE_TURN, winner_id="G1",
                               loser_id="G2", elo_before={"G1":1200,"G2":1200},
                               elo_after={"G1":1216.0,"G2":1184.0}, transcript="...", tick=1))
    return m

def test_elo_trajectory():
    traj = elo_trajectory(_mem())
    assert traj == [{"match": 1, "tick": 1, "best_elo": 1216.0}]

def test_write_results(tmp_path):
    out = str(tmp_path)
    write_results(_mem(), "Overview text.", out)
    for f in ["hypotheses.json","reviews.json","tournament.jsonl","elo_trajectory.csv",
              "research_overview.md","research_overview.json"]:
        assert os.path.isfile(os.path.join(out, f))
    assert "Overview text." in open(os.path.join(out,"research_overview.md")).read()
    hyps = json.load(open(os.path.join(out,"hypotheses.json")))
    assert hyps[0]["id"] == "G1"   # highest Elo first
    assert "match,tick,best_elo" in open(os.path.join(out,"elo_trajectory.csv")).read()

def test_summary_line():
    assert "hypotheses" in summary_line(_mem())


def test_grounding_stats_counts_grounded_reviews():
    m = ContextMemory(research_plan=ResearchPlan(goal="g"))
    m.add_hypothesis(Hypothesis(id="G1", text="t", title="T", source_strategy="s"))
    m.add_hypothesis(Hypothesis(id="G2", text="t", title="T", source_strategy="s"))
    m.add_review(Review(hypothesis_id="G1", type="full", text="ok", tool_grounded=True))
    m.add_review(Review(hypothesis_id="G1", type="deep_verification", text="ok", tool_grounded=False))
    m.add_review(Review(hypothesis_id="G2", type="full", text="ok", tool_grounded=False))
    s = grounding_stats(m)
    assert s == {"reviews_total": 3, "reviews_grounded": 1,
                 "hypotheses_total": 2, "hypotheses_grounded": 1}
    assert grounding_line(m) == "grounding: 1/3 reviews grounded"


def test_write_results_records_grounding(tmp_path):
    out = str(tmp_path)
    write_results(_mem(), "Overview text.", out)
    ov = json.load(open(os.path.join(out, "research_overview.json")))
    # _mem has one ungrounded full review -> 0/1 grounded, visible in the saved record
    assert ov["grounding"] == {"reviews_total": 1, "reviews_grounded": 0,
                               "hypotheses_total": 2, "hypotheses_grounded": 0}
