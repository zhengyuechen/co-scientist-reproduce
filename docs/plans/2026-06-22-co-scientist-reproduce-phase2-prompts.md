# Co-Scientist Reproduction — Phase 2 (Prompts) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build the prompt layer — the verbatim SN9 prompts, the reconstructed (SM-gap) prompts, and a render layer that fills placeholders — so Phase 3 agents have all their prompts.

**Architecture:** Three files under `cosci/prompts/`. `verbatim.py` holds byte-faithful SN9 prompts + a `PLACEHOLDER_MAP`. `reconstructed.py` holds in-SN9-style prompts for the gaps the paper didn't publish, each tagged `RECONSTRUCTED`. `render.py` substitutes placeholders (handling the supplement's space-containing names) and assembles meta-review feedback into the `{instructions}`/`{notes}` slots.

**Tech Stack:** Python 3.11 (conda env `cosci-reproduce`), stdlib `re`, `pytest`.

**Plan series:** Plan 2 of 6. Builds on Phase 1 (merged to `main`). Source-of-truth for verbatim prompts is the committed design spec, Appendix A: `docs/2026-06-22-co-scientist-reproduce-design.md`.

## Global Constraints

- Python 3.11, env `cosci-reproduce`; tests via `conda run -n cosci-reproduce pytest`.
- Repo `/Users/jeremychen/My Drive/Ran's Lab/projects/research-agents/co-scientist-reproduce`, already a git repo. Work on a branch `phase-2-prompts` (NOT main).
- **Provenance discipline (hard):** `verbatim.py` prompts must be byte-faithful to spec Appendix A (which is byte-faithful to SN9), *including* the supplement's inconsistent placeholder spellings (`{hypothesis 1}`, `{review 1}`, and `{review1}` vs `{review 2}`). `reconstructed.py` prompts are authored in SN9 style and EACH must carry a `# RECONSTRUCTED — not from SM. Methods intent: <…>` comment.
- **No literal braces** in any prompt string except placeholders: reconstructed prompts must describe JSON output in words (e.g. "return a JSON object with keys 'x' and 'y'"), never show literal `{...}` JSON — so the render regex never mistakes JSON for a placeholder.
- Commit messages plain — NO "Co-Authored-By"/"Claude"/attribution trailer.
- TDD; DRY; YAGNI.

## Canonical placeholders (shared vocabulary)

Canonical arg names used across prompts: `goal`, `preferences`, `instructions`, `idea_attributes`, `reviews_overview`, `articles_with_reasoning`, `source_hypothesis`, `transcript`, `article`, `hypothesis`, `notes`, `hypothesis_1`, `hypothesis_2`, `review_1`, `review_2`, `hypotheses`, `reviews`, `research_overview`, `goal_raw`.

`PLACEHOLDER_MAP` (verbatim spelling → canonical) — only the space/irregular ones need mapping; everything else maps to itself:
```python
PLACEHOLDER_MAP = {
    "hypothesis 1": "hypothesis_1",
    "hypothesis 2": "hypothesis_2",
    "review 1": "review_1",
    "review 2": "review_2",
    "review1": "review_1",
}
```

---

## File structure (Phase 2)

| File | Responsibility |
|---|---|
| `cosci/prompts/__init__.py` | Package marker; re-export the prompt namespaces |
| `cosci/prompts/verbatim.py` | 8 SM-verbatim SN9 prompts + `PLACEHOLDER_MAP` |
| `cosci/prompts/reconstructed.py` | 16 reconstructed prompts (SN9-style, tagged) |
| `cosci/prompts/render.py` | `render()` + `assemble_instructions()` + `KNOWN_PLACEHOLDERS` |
| `tests/test_prompts_verbatim.py` | placeholder presence + map coverage |
| `tests/test_prompts_reconstructed.py` | presence + required placeholders + no literal braces |
| `tests/test_render.py` | substitution, space-placeholder handling, missing-var raise, feedback injection |

---

## Task 1: Verbatim SN9 prompts

**Files:**
- Create: `cosci/prompts/__init__.py`, `cosci/prompts/verbatim.py`
- Test: `tests/test_prompts_verbatim.py`

**Interfaces:**
- Produces 8 module-level `str` constants in `verbatim.py`, each transcribed byte-faithfully from the design spec Appendix A section noted:
  - `GEN_LITERATURE` ← Appendix A.1 (SN9.1 #1)
  - `GEN_DEBATE` ← A.2 (SN9.1 #2)
  - `REFLECT_OBSERVATION` ← A.3 (SN9.2)
  - `RANK_SINGLE_TURN` ← A.4 (SN9.3 #1)
  - `RANK_MULTI_TURN` ← A.5 (SN9.3 #2)
  - `EVO_FEASIBILITY` ← A.6 (SN9.4 #1)
  - `EVO_OUT_OF_BOX` ← A.7 (SN9.4 #2)
  - `META_SYNTHESIS` ← A.8 (SN9.5)
- Plus `PLACEHOLDER_MAP` (exactly as in "Canonical placeholders" above).
- Source to copy from: `docs/2026-06-22-co-scientist-reproduce-design.md`, Appendix A (already in this repo). Copy the prompt body inside each code block verbatim, preserving `{placeholders}` exactly (including `{hypothesis 1}`, `{review1}`, etc.).

- [ ] **Step 1: Write the failing test**

`tests/test_prompts_verbatim.py`:
```python
from cosci.prompts import verbatim as V

def test_all_eight_prompts_present_and_nonempty():
    for name in ["GEN_LITERATURE", "GEN_DEBATE", "REFLECT_OBSERVATION",
                 "RANK_SINGLE_TURN", "RANK_MULTI_TURN", "EVO_FEASIBILITY",
                 "EVO_OUT_OF_BOX", "META_SYNTHESIS"]:
        assert isinstance(getattr(V, name), str) and len(getattr(V, name)) > 100

def test_key_placeholders_present():
    assert "{goal}" in V.GEN_LITERATURE
    assert "{articles_with_reasoning}" in V.GEN_LITERATURE
    assert "{idea_attributes}" in V.GEN_DEBATE and "{transcript}" in V.GEN_DEBATE
    assert "{article}" in V.REFLECT_OBSERVATION and "{hypothesis}" in V.REFLECT_OBSERVATION
    # supplement's inconsistent spacing preserved verbatim:
    assert "{hypothesis 1}" in V.RANK_SINGLE_TURN and "{review 1}" in V.RANK_SINGLE_TURN
    assert "{review1}" in V.RANK_MULTI_TURN and "{review 2}" in V.RANK_MULTI_TURN
    assert "{hypotheses}" in V.EVO_OUT_OF_BOX
    assert "{reviews}" in V.META_SYNTHESIS

def test_placeholder_map_covers_space_names():
    assert V.PLACEHOLDER_MAP["hypothesis 1"] == "hypothesis_1"
    assert V.PLACEHOLDER_MAP["review1"] == "review_1"
    assert V.PLACEHOLDER_MAP["review 2"] == "review_2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n cosci-reproduce pytest tests/test_prompts_verbatim.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cosci.prompts'`.

- [ ] **Step 3: Write minimal implementation**

Create `cosci/prompts/__init__.py` (empty).
Create `cosci/prompts/verbatim.py`: open the design spec, Appendix A, and copy each prompt's body (the text inside each ```code fence```) into the corresponding constant as a triple-quoted string. Header the file:
```python
"""SM-verbatim SN9 prompts. Byte-faithful to the supplement (via design spec Appendix A),
including the supplement's own inconsistent placeholder spellings. Do not 'fix' them here —
normalization happens at render time via PLACEHOLDER_MAP."""

GEN_LITERATURE = """<paste A.1 body verbatim>"""
# ... GEN_DEBATE (A.2), REFLECT_OBSERVATION (A.3), RANK_SINGLE_TURN (A.4),
#     RANK_MULTI_TURN (A.5), EVO_FEASIBILITY (A.6), EVO_OUT_OF_BOX (A.7), META_SYNTHESIS (A.8)

PLACEHOLDER_MAP = {
    "hypothesis 1": "hypothesis_1",
    "hypothesis 2": "hypothesis_2",
    "review 1": "review_1",
    "review 2": "review_2",
    "review1": "review_1",
}
```
Each `# SN9.x` citation comment above its constant.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n cosci-reproduce pytest tests/test_prompts_verbatim.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add cosci/prompts/__init__.py cosci/prompts/verbatim.py tests/test_prompts_verbatim.py
git commit -m "feat(prompts): verbatim SN9 prompts + placeholder map"
```

---

## Task 2: Reconstructed (SM-gap) prompts

**Files:**
- Create: `cosci/prompts/reconstructed.py`
- Test: `tests/test_prompts_reconstructed.py`

**Interfaces:**
- Produces 16 `str` constants, each authored in SN9 style (mirror the structure/tone of the verbatim prompts), each preceded by `# RECONSTRUCTED — not from SM. Methods intent: <…>`. No literal `{}` braces except placeholders. Each must contain the placeholders listed.

| Constant | Methods intent (one line) | Required placeholders | Style anchor |
|---|---|---|---|
| `GEN_ITERATIVE_ASSUMPTIONS` | Generate a hypothesis by identifying testable intermediate assumptions via conditional reasoning hops, then aggregating | `{goal}`, `{preferences}`, `{instructions}` | GEN_LITERATURE |
| `GEN_RESEARCH_EXPANSION` | Propose hypotheses in under-explored areas, informed by existing hypotheses + meta-review overview | `{goal}`, `{preferences}`, `{research_overview}` | GEN_LITERATURE |
| `REFLECT_INITIAL` | Fast tool-free review: correctness, quality, novelty, safety; flag clearly-flawed/unsafe for discard | `{goal}`, `{hypothesis}` | REFLECT_OBSERVATION |
| `REFLECT_FULL` | Full review with literature grounding: novelty + correctness, return assessments + a safety verdict + references | `{goal}`, `{hypothesis}`, `{articles_with_reasoning}` | REFLECT_OBSERVATION |
| `REFLECT_DEEP_VERIFICATION` | Decompose hypothesis into core + sub-assumptions, evaluate each independently for plausibility | `{hypothesis}` | REFLECT_OBSERVATION |
| `REFLECT_SIMULATION` | Step-wise simulate the hypothesis's mechanism/experiment to surface failure modes | `{hypothesis}` | REFLECT_OBSERVATION |
| `REFLECT_RECURRENT` | Re-review adapting to tournament results + meta-review feedback | `{hypothesis}`, `{notes}` | REFLECT_OBSERVATION |
| `EVO_COMBINE` | Combine the best parts of two top hypotheses into a stronger one | `{goal}`, `{preferences}`, `{hypotheses}` | EVO_OUT_OF_BOX |
| `EVO_SIMPLIFY` | Refine a hypothesis to be simpler and more testable | `{goal}`, `{preferences}`, `{hypothesis}` | EVO_FEASIBILITY |
| `EVO_GROUNDING` | Strengthen a hypothesis by grounding it in retrieved literature | `{goal}`, `{preferences}`, `{hypothesis}`, `{articles_with_reasoning}` | EVO_FEASIBILITY |
| `EVO_INSPIRATION` | Generate a new hypothesis inspired by (not copied from) existing ones | `{goal}`, `{preferences}`, `{hypotheses}` | EVO_OUT_OF_BOX |
| `META_RESEARCH_OVERVIEW` | Synthesize top hypotheses into a coherent research overview: directions + justifications | `{goal}`, `{hypotheses}` | META_SYNTHESIS |
| `META_SYSTEM_FEEDBACK` | Analyze all reviews + debate transcripts → common strengths/weaknesses as system-wide feedback | `{goal}`, `{reviews}` | META_SYNTHESIS |
| `SUP_GOAL_PARSE` | Parse a natural-language research goal into preferences, attributes, constraints; return a JSON object (described in words) | `{goal_raw}` | (concise, structured) |
| `SUP_SAFETY_REVIEW_GOAL` | Safety-review a research goal; conclude safe/unsafe with reason | `{goal_raw}` | REFLECT_OBSERVATION |
| `SUP_SAFETY_REVIEW_HYPOTHESIS` | Safety-review a hypothesis; conclude safe/unsafe with reason | `{goal}`, `{hypothesis}` | REFLECT_OBSERVATION |

- [ ] **Step 1: Write the failing test**

`tests/test_prompts_reconstructed.py`:
```python
import re
from cosci.prompts import reconstructed as R

REQUIRED = {
    "GEN_ITERATIVE_ASSUMPTIONS": ["{goal}", "{preferences}", "{instructions}"],
    "GEN_RESEARCH_EXPANSION": ["{goal}", "{preferences}", "{research_overview}"],
    "REFLECT_INITIAL": ["{goal}", "{hypothesis}"],
    "REFLECT_FULL": ["{goal}", "{hypothesis}", "{articles_with_reasoning}"],
    "REFLECT_DEEP_VERIFICATION": ["{hypothesis}"],
    "REFLECT_SIMULATION": ["{hypothesis}"],
    "REFLECT_RECURRENT": ["{hypothesis}", "{notes}"],
    "EVO_COMBINE": ["{goal}", "{preferences}", "{hypotheses}"],
    "EVO_SIMPLIFY": ["{goal}", "{preferences}", "{hypothesis}"],
    "EVO_GROUNDING": ["{goal}", "{preferences}", "{hypothesis}", "{articles_with_reasoning}"],
    "EVO_INSPIRATION": ["{goal}", "{preferences}", "{hypotheses}"],
    "META_RESEARCH_OVERVIEW": ["{goal}", "{hypotheses}"],
    "META_SYSTEM_FEEDBACK": ["{goal}", "{reviews}"],
    "SUP_GOAL_PARSE": ["{goal_raw}"],
    "SUP_SAFETY_REVIEW_GOAL": ["{goal_raw}"],
    "SUP_SAFETY_REVIEW_HYPOTHESIS": ["{goal}", "{hypothesis}"],
}

def test_all_constants_present_nonempty():
    for name in REQUIRED:
        assert isinstance(getattr(R, name), str) and len(getattr(R, name)) > 80

def test_required_placeholders_present():
    for name, phs in REQUIRED.items():
        body = getattr(R, name)
        for ph in phs:
            assert ph in body, f"{name} missing {ph}"

def test_no_literal_json_braces():
    # the only legal {..} tokens are known placeholders; no literal JSON braces
    known = {ph.strip("{}") for phs in REQUIRED.values() for ph in phs}
    for name in REQUIRED:
        for tok in re.findall(r"\{([^{}]*)\}", getattr(R, name)):
            assert tok in known, f"{name} has unexpected brace token: {{{tok}}}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n cosci-reproduce pytest tests/test_prompts_reconstructed.py -v`
Expected: FAIL with `ModuleNotFoundError` / `AttributeError`.

- [ ] **Step 3: Write minimal implementation**

Create `cosci/prompts/reconstructed.py`. Module docstring noting all prompts are RECONSTRUCTED (not from SM), authored in SN9 style to fill gaps the paper described in Methods but did not publish as prompts. For each of the 16 constants: a `# RECONSTRUCTED — not from SM. Methods intent: <intent>` comment, then the prompt authored in SN9 style — open with an expert-role sentence, state the Goal/Criteria, give clear numbered instructions, end with an explicit output instruction (and for safety/observation-style prompts, a required concluding line, e.g. `conclude with "safety: <safe or unsafe>" and a reason`). Use ONLY the listed placeholders; describe any JSON output in words (no literal braces).

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n cosci-reproduce pytest tests/test_prompts_reconstructed.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add cosci/prompts/reconstructed.py tests/test_prompts_reconstructed.py
git commit -m "feat(prompts): reconstructed SM-gap prompts (SN9 style, tagged)"
```

---

## Task 3: Render layer

**Files:**
- Create: `cosci/prompts/render.py`
- Modify: `cosci/prompts/__init__.py` (re-export)
- Test: `tests/test_render.py`

**Interfaces:**
- Produces:
  - `KNOWN_PLACEHOLDERS: set[str]` — all canonical arg names (the union from the "Canonical placeholders" list).
  - `render(template: str, **vars) -> str` — replaces every `{token}` whose mapped canonical name is in `KNOWN_PLACEHOLDERS`; maps via `PLACEHOLDER_MAP` (default identity); raises `KeyError` if a known placeholder in the template has no provided value; leaves unknown `{token}` literal (defensive, though prompts avoid literal braces).
  - `assemble_instructions(memory) -> str` — returns `memory.system_feedback` if non-empty else `""`; this is the value callers pass as `instructions=`/`notes=` so meta-review feedback propagates into prompts (the paper's feedback loop).

- [ ] **Step 1: Write the failing test**

`tests/test_render.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n cosci-reproduce pytest tests/test_render.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cosci.prompts.render'`.

- [ ] **Step 3: Write minimal implementation**

`cosci/prompts/render.py`:
```python
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
```
Update `cosci/prompts/__init__.py` to re-export:
```python
from cosci.prompts import verbatim, reconstructed
from cosci.prompts.render import render, assemble_instructions, KNOWN_PLACEHOLDERS
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n cosci-reproduce pytest tests/test_render.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Run full suite + commit**

Run: `conda run -n cosci-reproduce pytest -q`
Expected: all Phase 1 + Phase 2 tests pass.

```bash
git add cosci/prompts/render.py cosci/prompts/__init__.py tests/test_render.py
git commit -m "feat(prompts): render layer + feedback assembly"
```

---

## Phase 2 done-criteria
- `conda run -n cosci-reproduce pytest -q` green.
- `cosci.prompts` exposes `verbatim` (8 prompts + `PLACEHOLDER_MAP`), `reconstructed` (16 tagged prompts), `render`, `assemble_instructions`, `KNOWN_PLACEHOLDERS`.
- Every reconstructed prompt carries its `# RECONSTRUCTED` provenance comment (verified in review).

## Self-review notes
- **Spec coverage:** spec §6 (prompts policy, verbatim vs reconstructed, PLACEHOLDER_MAP, render) → Tasks 1–3; Appendix A → Task 1; Appendix C inventory → Task 2.
- **No placeholders:** verbatim source is the committed spec Appendix A (named per constant); reconstructed prompts have a per-prompt spec (intent + placeholders + style anchor); render code is complete.
- **Type consistency:** `PLACEHOLDER_MAP` defined in `verbatim.py`, consumed by `render.py`; `KNOWN_PLACEHOLDERS` canonical names match the placeholders the prompts use and the render tests.
