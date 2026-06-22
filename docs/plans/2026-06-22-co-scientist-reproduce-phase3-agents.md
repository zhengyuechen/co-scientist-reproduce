# Co-Scientist Reproduction — Phase 3 (Agents) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build the six specialized agents (Generation, Reflection, Ranking, Evolution, Proximity, Meta-review) on a shared `Agent`/`Results` contract, each wired to the Phase 2 prompts and Phase 1 primitives, each tested deterministically against `FakeLLM`.

**Architecture:** Each agent exposes `async execute(task, memory, llm, cfg) -> Results`. An agent reads/writes `ContextMemory`, calls the LLM via `LLMClient`, renders prompts via `cosci.prompts.render`, and **returns** outputs plus follow-up `Task` requests (it never enqueues directly — the Supervisor enqueues in Phase 4). Grounding (web/arXiv) is NOT wired here; generation/reflection render their prompts with an empty `articles_with_reasoning` and rely on the model's parametric knowledge (Phase 5 injects real grounding). This keeps Phase 3 self-contained and testable with no network.

**Tech Stack:** Python 3.11 (env `cosci-reproduce`), stdlib `re`/`dataclasses`, `sentence-transformers` (proximity only, injectable for tests), `pytest`/`pytest-asyncio`.

**Plan series:** Plan 3 of 6. Builds on Phases 1–2 (merged to `main`).

## Global Constraints

- Python 3.11, env `cosci-reproduce`; tests via `conda run -n cosci-reproduce pytest`.
- Repo `/Users/jeremychen/My Drive/Ran's Lab/projects/research-agents/co-scientist-reproduce`, a git repo. Work on branch `phase-3-agents` (NOT main).
- Agents return `Results`; they do NOT mutate the task queue. Follow-up chaining (per SN8): new/evolved hypothesis → REVIEW_HYPOTHESIS; review done → ADD_TO_TOURNAMENT.
- **Provenance:** Elo seed is `cosci.elo.INITIAL_ELO` (=1200, SM). Match-mode rule (top-ranked → multi-turn debate; else single-turn) is SM (Methods p.26). Evolution top-K=5, overview top-N=10 are SM (SN8). All other choices (selection heuristics, parsing) are OURS — comment them.
- **Safety:** Reflection parses a `safety:` verdict and sets `hypothesis.safety` + `review.safety`. Unsafe hypotheses are NOT excluded by the agent itself — the agent still emits the ADD_TO_TOURNAMENT follow-up; the Supervisor (Phase 4) drops it. `ContextMemory.active_hypotheses()` already filters UNSAFE, so ranking/evolution/proximity/meta naturally skip them.
- Commit messages plain — NO attribution trailer. TDD; DRY; YAGNI.

## Shared interfaces (defined in Task 1, consumed by all)

- `Results` (dataclass): `new_hypotheses: list[Hypothesis]=[]`, `reviews: list[Review]=[]`, `match: MatchResult|None=None`, `follow_ups: list[Task]=[]`, `overview: str|None=None`, `feedback: str|None=None`.
- `Agent` (Protocol): `async execute(self, task: Task, memory: ContextMemory, llm: LLMClient, cfg: Config) -> Results`.
- `parse_label(text, *labels) -> str|None`: extract the value of the last `label: <value>` marker (case-insensitive), stripping `<>`; tries each label in order. Used for `better idea`/`better hypothesis`, `safety`, `verification`, `decision`, `simulation`, `assessment`.

---

## File structure (Phase 3)

| File | Responsibility |
|---|---|
| `cosci/agents/__init__.py` | package marker + re-export agents |
| `cosci/agents/base.py` | `Results`, `Agent` protocol, `parse_label` |
| `cosci/agents/generation.py` | `GenerationAgent` — strategies → new hypotheses |
| `cosci/agents/reflection.py` | `ReflectionAgent` — full + deep-verification reviews + safety verdict |
| `cosci/agents/ranking.py` | `RankingAgent` — AddToTournament + RunTournamentBatch (Elo debate) |
| `cosci/agents/evolution.py` | `EvolutionAgent` — combine top hypotheses |
| `cosci/agents/proximity.py` | `ProximityAgent` — embedding similarity graph (injectable encoder) |
| `cosci/agents/meta_review.py` | `MetaReviewAgent` — system feedback + final overview |
| `tests/test_agent_*.py` | one per agent, against FakeLLM |

---

## Task 1: Agent contract (base.py)

**Files:** Create `cosci/agents/__init__.py`, `cosci/agents/base.py`; Test `tests/test_agent_base.py`

**Interfaces:** Produces `Results`, `Agent`, `parse_label` (signatures above).

- [ ] **Step 1: Write the failing test** — `tests/test_agent_base.py`:
```python
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
```

- [ ] **Step 2: Run test, expect FAIL** — `conda run -n cosci-reproduce pytest tests/test_agent_base.py -v` → `ModuleNotFoundError`.

- [ ] **Step 3: Implement** — `cosci/agents/__init__.py` (empty for now); `cosci/agents/base.py`:
```python
"""Shared agent contract: Results payload, Agent protocol, and a verdict-line parser."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Protocol
from cosci.models import Hypothesis, Review, MatchResult, Task
from cosci.memory import ContextMemory
from cosci.config import Config
from cosci.llm import LLMClient

@dataclass
class Results:
    new_hypotheses: list[Hypothesis] = field(default_factory=list)
    reviews: list[Review] = field(default_factory=list)
    match: MatchResult | None = None
    follow_ups: list[Task] = field(default_factory=list)
    overview: str | None = None
    feedback: str | None = None

class Agent(Protocol):
    async def execute(self, task: Task, memory: ContextMemory,
                      llm: LLMClient, cfg: Config) -> Results: ...

def parse_label(text: str, *labels: str) -> str | None:
    """Return the value of the LAST `label: <value>` marker (case-insensitive), or None.
    Strips surrounding <> and trailing prose after the value token-run."""
    best = None
    for label in labels:
        for m in re.finditer(rf"{re.escape(label)}\s*:\s*<?\s*([A-Za-z0-9 _\-]+?)\s*>?(?:\s|$|[.,;])",
                             text, re.IGNORECASE):
            best = m.group(1).strip().lower()
    return best
```

- [ ] **Step 4: Run test, expect PASS.**
- [ ] **Step 5: Commit** — `git add cosci/agents/__init__.py cosci/agents/base.py tests/test_agent_base.py && git commit -m "feat(agents): Results/Agent contract + verdict parser"`

---

## Task 2: Generation agent

**Files:** Create `cosci/agents/generation.py`; Test `tests/test_agent_generation.py`

**Behavior:** `GenerationAgent(strategies: list[str] | None = None)`. `execute` handles `TaskType.CREATE_INITIAL_HYPOTHESES`. Default strategies `["literature_review", "scientific_debate"]` (the two with verbatim prompts; `iterative_assumptions`/`research_expansion` selectable). For each strategy: render its prompt (verbatim for the first two; reconstructed for the others) with `goal`, `preferences`, and empty `instructions`/`articles_with_reasoning`/`source_hypothesis`/`idea_attributes`/`research_overview` as needed (use `assemble_instructions(memory)` for `{instructions}`); one LLM call (agent="generation"); the response is the hypothesis text. For `scientific_debate`, if the response contains `HYPOTHESIS`, take the text after the last `HYPOTHESIS` marker. Title = first non-empty line truncated to ~80 chars (OURS). Create `Hypothesis(id=memory.new_id("G"), title, text, source_strategy=strategy, origin=Origin.GENERATED, created_tick=memory.tick)`, add to memory, and emit `Task(AgentName.REFLECTION, TaskType.REVIEW_HYPOTHESIS, target_id=hid)`.

**Interfaces:** Consumes `prompts.verbatim/reconstructed`, `render`, `assemble_instructions`, `memory`, `Origin`. Produces `GenerationAgent`.

- [ ] **Step 1: Failing test** — `tests/test_agent_generation.py`:
```python
import pytest
from cosci.agents.generation import GenerationAgent
from cosci.agents.base import Results
from cosci.memory import ContextMemory
from cosci.models import ResearchPlan, Task, AgentName, TaskType
from cosci.config import load_config
from tests.fake_llm import FakeLLM

def _router(agent, messages):
    # debate returns a HYPOTHESIS-marked response; others plain prose
    content = messages[-1]["content"]
    if "simulated discussion" in content or "collaborative discourse" in content:
        return "...debate...\nHYPOTHESIS\nNovel mechanism A explains the effect."
    return "Hypothesis: mechanism B drives the observation in detail."

@pytest.mark.asyncio
async def test_generation_makes_one_hypothesis_per_strategy(tmp_path, monkeypatch):
    cfg = load_config("config.yaml")
    mem = ContextMemory(research_plan=ResearchPlan(goal="cure X", preferences=["novel"]))
    agent = GenerationAgent(strategies=["literature_review", "scientific_debate"])
    task = Task(agent=AgentName.GENERATION, action=TaskType.CREATE_INITIAL_HYPOTHESES)
    res = await agent.execute(task, mem, FakeLLM(_router), cfg)
    assert len(res.new_hypotheses) == 2
    assert {h.source_strategy for h in res.new_hypotheses} == {"literature_review", "scientific_debate"}
    # debate hypothesis text taken after the HYPOTHESIS marker
    deb = [h for h in res.new_hypotheses if h.source_strategy == "scientific_debate"][0]
    assert deb.text.strip().startswith("Novel mechanism A")
    # each is in memory and each emits a reflection follow-up
    assert all(mem.get(h.id) is not None for h in res.new_hypotheses)
    assert len(res.follow_ups) == 2
    assert all(f.action == TaskType.REVIEW_HYPOTHESIS for f in res.follow_ups)
```

- [ ] **Step 2: FAIL.** **Step 3: Implement** per Behavior (render prompts, parse, build hypotheses, follow-ups). Use `cfg` only for the LLM model routing (handled inside the client). **Step 4: PASS.** **Step 5: Commit** `feat(agents): generation agent (strategy-per-hypothesis)`.

---

## Task 3: Reflection agent

**Files:** Create `cosci/agents/reflection.py`; Test `tests/test_agent_reflection.py`

**Behavior:** `execute` handles `REVIEW_HYPOTHESIS` (target_id=hid). Runs two reviews on the hypothesis text: `REFLECT_FULL` (render with `goal`, `hypothesis`, empty `articles_with_reasoning`) and `REFLECT_DEEP_VERIFICATION` (render with `hypothesis`). For each: one LLM call (agent="reflection"); build `Review(hypothesis_id=hid, type=<"full"|"deep_verification">, text=<response>, tool_grounded=False)`. From the FULL review response, parse `safety:` via `parse_label(...,"safety")`; if "unsafe" → set that review's `safety=Safety.UNSAFE` and set `hypothesis.safety=Safety.UNSAFE` (+ `safety_reason` = the response's safety line); else `Safety.SAFE`. Append reviews to memory (`memory.add_review`). Emit `Task(AgentName.RANKING, TaskType.ADD_TO_TOURNAMENT, target_id=hid)` (Supervisor will drop it later if unsafe). Return `Results(reviews=[...], follow_ups=[...])`.

- [ ] **Step 1: Failing test** — `tests/test_agent_reflection.py`:
```python
import pytest
from cosci.agents.reflection import ReflectionAgent
from cosci.memory import ContextMemory
from cosci.models import Hypothesis, ResearchPlan, Task, AgentName, TaskType, Safety
from cosci.config import load_config
from tests.fake_llm import FakeLLM

def _safe_router(agent, messages):
    c = messages[-1]["content"]
    if "deep verification" in c: return "assumptions ok\nverification: verified"
    return "novel and correct\nsafety: safe"

def _unsafe_router(agent, messages):
    c = messages[-1]["content"]
    if "deep verification" in c: return "verification: uncertain"
    return "concerning dual-use\nsafety: unsafe because it enables harm"

async def _run(router):
    cfg = load_config("config.yaml")
    mem = ContextMemory(research_plan=ResearchPlan(goal="g"))
    h = Hypothesis(id="G1", text="some hypothesis", title="T", source_strategy="s")
    mem.add_hypothesis(h)
    res = await ReflectionAgent().execute(
        Task(agent=AgentName.REFLECTION, action=TaskType.REVIEW_HYPOTHESIS, target_id="G1"),
        mem, FakeLLM(router), cfg)
    return mem, res

@pytest.mark.asyncio
async def test_reflection_safe_path():
    mem, res = await _run(_safe_router)
    assert len(res.reviews) == 2 and len(mem.reviews["G1"]) == 2
    assert mem.get("G1").safety == Safety.SAFE
    assert res.follow_ups[0].action == TaskType.ADD_TO_TOURNAMENT

@pytest.mark.asyncio
async def test_reflection_unsafe_sets_quarantine_flag():
    mem, res = await _run(_unsafe_router)
    assert mem.get("G1").safety == Safety.UNSAFE
    assert any(r.safety == Safety.UNSAFE for r in res.reviews)
    # agent still emits the follow-up; Supervisor (Phase 4) is what drops it
    assert res.follow_ups[0].action == TaskType.ADD_TO_TOURNAMENT
```

- [ ] Steps 2–4 (FAIL→implement→PASS). **Step 5: Commit** `feat(agents): reflection agent (full + deep-verification + safety verdict)`.

---

## Task 4: Ranking agent

**Files:** Create `cosci/agents/ranking.py`; Test `tests/test_agent_ranking.py`

**Behavior:**
- `ADD_TO_TOURNAMENT` (target_id=hid): if `hypothesis.elo_rating is None`, set it to `cosci.elo.INITIAL_ELO`. Idempotent (skip if already set). Return empty `Results`.
- `RUN_TOURNAMENT_BATCH`: from `memory.active_hypotheses()` that have an `elo_rating`, select a pair. **Selection (OURS):** sort by `(matches_played, -elo_rating)`; take the two least-played. **Mode (SM):** `multi_turn` if BOTH are in the top half by Elo, else `single_turn`. Render `RANK_MULTI_TURN`/`RANK_SINGLE_TURN` with `goal`, `preferences`, the two hypotheses' texts (→ `hypothesis_1`/`hypothesis_2`), their latest review texts (→ `review_1`/`review_2`, "" if none), empty `notes`/`idea_attributes`. One LLM call (agent="ranking"). Parse winner via `parse_label(resp, "better idea", "better hypothesis")` → "1"/"2"; default to hypothesis_1 on parse failure (log). Compute new Elos via `cosci.elo.update`. Update both hypotheses' `elo_rating`, increment both `matches_played`. Build + record `MatchResult` (mode, winner/loser ids, elo_before/after, transcript=resp, tick=memory.tick). Return `Results(match=...)`.

- [ ] **Step 1: Failing test** — `tests/test_agent_ranking.py`:
```python
import pytest
from cosci.agents.ranking import RankingAgent
from cosci.elo import INITIAL_ELO
from cosci.memory import ContextMemory
from cosci.models import Hypothesis, ResearchPlan, Task, AgentName, TaskType, DebateMode
from cosci.config import load_config
from tests.fake_llm import FakeLLM

def _mk(mem, hid):
    h = Hypothesis(id=hid, text=f"text {hid}", title=hid, source_strategy="s")
    mem.add_hypothesis(h); return h

@pytest.mark.asyncio
async def test_add_to_tournament_sets_initial_elo_idempotent():
    cfg = load_config("config.yaml"); mem = ContextMemory()
    _mk(mem, "G1")
    agent = RankingAgent()
    await agent.execute(Task(agent=AgentName.RANKING, action=TaskType.ADD_TO_TOURNAMENT, target_id="G1"),
                        mem, FakeLLM(lambda a, m: ""), cfg)
    assert mem.get("G1").elo_rating == INITIAL_ELO
    mem.get("G1").elo_rating = 1400  # idempotent: second add must not reset
    await agent.execute(Task(agent=AgentName.RANKING, action=TaskType.ADD_TO_TOURNAMENT, target_id="G1"),
                        mem, FakeLLM(lambda a, m: ""), cfg)
    assert mem.get("G1").elo_rating == 1400

@pytest.mark.asyncio
async def test_run_batch_updates_elo_and_records_match():
    cfg = load_config("config.yaml")
    mem = ContextMemory(research_plan=ResearchPlan(goal="g"))
    for hid in ("G1", "G2"):
        _mk(mem, hid); mem.get(hid).elo_rating = INITIAL_ELO
    res = await RankingAgent().execute(
        Task(agent=AgentName.RANKING, action=TaskType.RUN_TOURNAMENT_BATCH),
        mem, FakeLLM(lambda a, m: "debate...\nbetter idea: 1"), cfg)
    assert res.match is not None
    win, lose = res.match.winner_id, res.match.loser_id
    assert mem.get(win).elo_rating > INITIAL_ELO and mem.get(lose).elo_rating < INITIAL_ELO
    assert mem.get(win).matches_played == 1 and len(mem.tournament) == 1
```

- [ ] Steps 2–4. **Step 5: Commit** `feat(agents): ranking agent (Elo tournament via debate)`.

---

## Task 5: Evolution agent

**Files:** Create `cosci/agents/evolution.py`; Test `tests/test_agent_evolution.py`

**Behavior:** `execute` handles `EVOLVE_TOP`. Take top-K (`cfg.evolution.top_k`, =5) of `memory.active_hypotheses()` sorted by `elo_rating` desc (None treated as INITIAL_ELO). If ≥2: render `EVO_COMBINE` with `goal`, `preferences`, and the top-2 hypotheses' texts joined (→ `hypotheses`). One LLM call (agent="evolution"). Response = evolved text; title derived (first line, ~80 chars). Create `Hypothesis(id=memory.new_id("E"), title, text, source_strategy="combine", origin=Origin.EVOLVED, parent_ids=[top[0].id, top[1].id], created_tick=memory.tick)`, add to memory. Emit `Task(REFLECTION, REVIEW_HYPOTHESIS, evolved_id)`. Return `Results(new_hypotheses=[evolved], follow_ups=[...])`. If <2 active: return empty `Results`.

- [ ] **Step 1: Failing test** — assert: with 3 active hypotheses (distinct Elo), evolution produces exactly 1 evolved hypothesis with `origin==EVOLVED`, `parent_ids` = the top 2 ids, source_strategy "combine", added to memory, and one REVIEW follow-up. With <2 active, returns no new hypotheses. (Use FakeLLM returning a fixed evolved-text string.)

- [ ] Steps 2–4. **Step 5: Commit** `feat(agents): evolution agent (combine top hypotheses)`.

---

## Task 6: Proximity agent

**Files:** Create `cosci/agents/proximity.py`; Test `tests/test_agent_proximity.py`

**Behavior:** `ProximityAgent(encoder=None)`. `encoder` is a callable `list[str] -> list[list[float]]` (or an object with `.encode`). If None, lazy-load `sentence-transformers` `cfg.proximity.model` on first use (do NOT import sentence-transformers at module top — import inside the lazy loader so tests need not load it). `execute` handles `UPDATE_PROXIMITY`: embed the texts of `memory.active_hypotheses()`, compute pairwise cosine similarity, and store `memory.proximity[id] = [{"other_id": j, "similarity": s}, ...]` for each. Return empty `Results`. No LLM call.

- [ ] **Step 1: Failing test** — inject a fake encoder mapping text→unit vector so identical texts → cosine 1.0; assert: two hypotheses with identical text get similarity ≈ 1.0 in `memory.proximity`; a third orthogonal one gets ≈ 0.0; the encoder was the injected one (no sentence-transformers import). Example fake encoder: map "a"→[1,0], "b"→[0,1], identical text → identical vector.

- [ ] Steps 2–4. **Step 5: Commit** `feat(agents): proximity agent (embedding similarity graph)`.

---

## Task 7: Meta-review agent

**Files:** Create `cosci/agents/meta_review.py`; Modify `cosci/agents/__init__.py` (re-export all six agents); Test `tests/test_agent_meta_review.py`

**Behavior:**
- `GENERATE_SYSTEM_FEEDBACK`: gather all review texts across `memory.reviews` (+ match transcripts from `memory.tournament`) into one block; render `META_SYSTEM_FEEDBACK` with `goal`, `reviews`=<block>. One LLM call (agent="meta_review"). Set `memory.system_feedback = response`. Return `Results(feedback=response)`.
- `GENERATE_FINAL_OVERVIEW`: take top-N (`cfg.overview.top_n`, =10) of `memory.active_hypotheses()` by Elo; render `META_RESEARCH_OVERVIEW` with `goal`, `hypotheses`=<joined texts>. One LLM call. Return `Results(overview=response)`.

- [ ] **Step 1: Failing test** — assert: SYSTEM_FEEDBACK writes `memory.system_feedback` and returns it in `Results.feedback`; FINAL_OVERVIEW returns `Results.overview` (non-empty) using the active hypotheses. Use FakeLLM returning fixed strings.

- [ ] **Step 2: FAIL. Step 3: implement + update `__init__.py` to re-export `GenerationAgent, ReflectionAgent, RankingAgent, EvolutionAgent, ProximityAgent, MetaReviewAgent, Results, Agent`. Step 4: PASS.**
- [ ] **Step 5: Run FULL suite** `conda run -n cosci-reproduce pytest -q` (Phases 1–3 all pass), then commit `feat(agents): meta-review agent + agents package exports`.

---

## Phase 3 done-criteria
- Full suite green; `cosci.agents` exposes the six agents + `Results`/`Agent`.
- Each agent: pure `execute(task, memory, llm, cfg) -> Results`; no direct queue mutation; safety verdict flows from reflection onto the hypothesis; Elo seeded from `INITIAL_ELO`; evolution/overview honor SM top-K/top-N.

## Self-review notes
- **Spec coverage:** spec §5.1–5.6 (the six agents) → Tasks 2–7; §5 Results/Agent contract → Task 1; safety flow (§4/§12) → Task 3 + memory filter; Elo seed/tournament (§7) → Task 4.
- **Deferred to later phases (intentional):** real grounding (Phase 5; here `articles_with_reasoning=""`); the Supervisor chaining/quarantine enforcement and DecideNextSteps (Phase 4); proximity-graph-guided match selection refinement (Phase 4 may revisit; Phase 3 uses the least-played heuristic).
- **Type consistency:** every agent `execute(task, memory, llm, cfg)`; `Results` field names reused; `parse_label`, `INITIAL_ELO`, `Origin`, `Safety`, `DebateMode` referenced as defined in Phase 1.
