"""Placeholder rendering. Handles the supplement's space-containing placeholder names
via PLACEHOLDER_MAP and injects meta-review feedback into the {instructions}/{notes} slots."""
from __future__ import annotations
import re
from cosci.prompts.verbatim import PLACEHOLDER_MAP

KNOWN_PLACEHOLDERS = {
    "goal", "preferences", "instructions", "idea_attributes", "reviews_overview",
    "articles_with_reasoning", "source_hypothesis", "transcript", "article",
    "hypothesis", "notes", "hypothesis_1", "hypothesis_2", "review_1", "review_2",
    "hypotheses", "reviews", "research_overview", "goal_raw",
}

_TOKEN = re.compile(r"\{([^{}]+)\}")

def render(template: str, **vars: object) -> str:
    def repl(m: re.Match) -> str:
        raw = m.group(1)
        canonical = PLACEHOLDER_MAP.get(raw, raw)
        if canonical not in KNOWN_PLACEHOLDERS:
            return m.group(0)  # leave unknown tokens literal
        if canonical not in vars:
            raise KeyError(f"missing template var '{canonical}' for placeholder '{{{raw}}}'")
        return str(vars[canonical])
    return _TOKEN.sub(repl, template)

def assemble_instructions(memory) -> str:
    return memory.system_feedback or ""
