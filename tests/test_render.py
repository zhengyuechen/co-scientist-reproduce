import pytest
from cosci.prompts.render import render, assemble_instructions, KNOWN_PLACEHOLDERS
from cosci.memory import ContextMemory

def test_render_simple_and_space_placeholder():
    tmpl = "Goal: {goal}\nH1: {hypothesis 1}\nR1: {review 1}"
    out = render(tmpl, goal="cure X", hypothesis_1="H-a", review_1="ok")
    assert "Goal: cure X" in out
    assert "H1: H-a" in out and "R1: ok" in out
    assert "{" not in out  # all placeholders filled

def test_render_missing_known_var_raises():
    with pytest.raises(KeyError):
        render("Goal: {goal}", )  # goal is a known placeholder, not provided

def test_render_leaves_unknown_token_literal():
    # an unknown brace token is left as-is (prompts avoid this, but be defensive)
    assert render("set X = {y_not_known}", goal="g") == "set X = {y_not_known}"

def test_assemble_instructions_returns_feedback():
    m = ContextMemory()
    assert assemble_instructions(m) == ""
    m.system_feedback = "avoid vague mechanisms"
    assert assemble_instructions(m) == "avoid vague mechanisms"
