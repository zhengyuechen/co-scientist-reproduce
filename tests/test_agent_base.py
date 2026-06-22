from cosci.agents.base import Results, parse_label
from cosci.models import Hypothesis

def test_results_defaults_independent():
    a, b = Results(), Results()
    a.new_hypotheses.append(Hypothesis(id="G1", text="t", title="T", source_strategy="s"))
    assert b.new_hypotheses == []  # no shared mutable default

def test_parse_label_picks_value_and_lowercases():
    assert parse_label("...\nbetter idea: 2", "better idea", "better hypothesis") == "2"
    assert parse_label("reasoning\nsafety: <UNSAFE> because x", "safety") == "unsafe"
    assert parse_label("no marker here", "safety") is None

def test_parse_label_takes_last_occurrence():
    assert parse_label("safety: safe\n...\nsafety: unsafe", "safety") == "unsafe"
