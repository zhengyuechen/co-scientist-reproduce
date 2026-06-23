# Co-Scientist Reproduction — Phase 4 (Orchestration) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build the `Supervisor` (SN8 task-chaining, safety gating, idle policy, termination) and the `Engine` (queue-driven loop, `continuous` + `round_based` modes, snapshots) that drive the six agents into a working hypothesis-evolution run.

**Architecture:** The `Engine` owns a `GlobalTaskQueue`; it seeds the first task from `Supervisor.start`, then loops: fetch a task → route to its agent → `execute` → `Supervisor.manage_follow_ups` enqueues follow-ups (dropping ADD_TO_TOURNAMENT for unsafe hypotheses) → when the queue idles, `Supervisor.decide_next_steps` enqueues tournament/evolution/proximity/feedback tasks → stop when `Supervisor.is_terminal`. Loop is **sequential** (faithful to SN8's `StartCoScientist`); an `asyncio.Lock` guards memory and `workers`>1 concurrent dispatch is a documented future enhancement.

**Tech Stack:** Python 3.11 (env `cosci-reproduce`), `asyncio`, `pytest`/`pytest-asyncio`.

**Plan series:** Plan 4 of 6. Builds on Phases 1–3 (merged to `main`).

## Global Constraints

- Python 3.11, env `cosci-reproduce`; tests via `conda run -n cosci-reproduce pytest` **run from the repo root** (agent/engine tests load `config.yaml` by relative path).
- Repo `/Users/jeremychen/My Drive/Ran's Lab/projects/research-agents/co-scientist-reproduce`, a git repo. Work on branch `phase-4-orchestration` (NOT main).
- **Faithful to SN8:** `manage_follow_ups` chaining = new/evolved→REVIEW, review→ADD_TO_TOURNAMENT; `decide_next_steps` = tournament batch (always), evolve (on plateau), periodic system-feedback/overview. Termination = `NumIdeas ≥ max_ideas` OR `min matches-per-idea ≥ max_matches` (SN8 loop guard) OR token budget.
- **Safety enforcement (the quarantine):** `manage_follow_ups` DROPS an ADD_TO_TOURNAMENT follow-up whose target hypothesis has `safety == UNSAFE` — this is where "unsafe hypotheses never enter the tournament" is actually enforced.
- Provenance: `INITIAL_ELO`/top-K/top-N/mode-rule are SM (already in agents). Sequential loop, lock, plateau heuristic, tick cadence are OURS — comment them.
- Commit messages plain — NO attribution trailer. TDD; DRY; YAGNI.

## File structure (Phase 4)

| File | Responsibility |
|---|---|
| `cosci/supervisor.py` | `Supervisor`: start, route, manage_follow_ups, summary_stats, is_terminal, decide_next_steps |
| `cosci/engine.py` | `run_engine(...)` — the queue-driven loop (continuous + round_based) + snapshots |
| `tests/test_supervisor.py` | unit tests per Supervisor method (FakeLLM) |
| `tests/test_engine.py` | end-to-end tiny run + round_based + unsafe-goal abort |

---

## Task 1: Supervisor

**Files:** Create `cosci/supervisor.py`; Test `tests/test_supervisor.py`

**Interfaces / behavior:** `Supervisor(agents: dict[AgentName, Agent])`.
- `async start(goal_raw, memory, llm, cfg) -> Task | None`:
  1. Safety goal-gate: `render(SUP_SAFETY_REVIEW_GOAL, goal_raw=goal_raw)` → `llm.complete("meta_review", ...)` → if `parse_label(resp,"safety")=="unsafe"` return `None` (run aborts).
  2. Parse: `render(SUP_GOAL_PARSE, goal_raw=goal_raw)` → LLM → `extract_json(resp)`; build `ResearchPlan(goal=data.get("goal", goal_raw), preferences=data.get("preferences",[]), attributes=data.get("attributes",[]), constraints=data.get("constraints",[]))`. On parse failure (ValueError), fall back to `ResearchPlan(goal=goal_raw)`.
  3. `memory.research_plan = plan`. Return `Task(AgentName.GENERATION, TaskType.CREATE_INITIAL_HYPOTHESES)`.
- `route(task) -> Agent`: `return self.agents[task.agent]`.
- `manage_follow_ups(results, memory) -> list[Task]`: for each `ft` in `results.follow_ups`: if `ft.action == ADD_TO_TOURNAMENT` and `memory.get(ft.target_id)` has `safety == Safety.UNSAFE` → skip (quarantine); else keep. Return kept list.
- `summary_stats(memory) -> dict`: `in_tourn = [h for h in memory.active_hypotheses() if h.elo_rating is not None]`; return `{"num_ideas": len(memory.active_hypotheses()), "num_in_tournament": len(in_tourn), "matches_total": len(memory.tournament), "min_matches_per_idea": min((h.matches_played for h in in_tourn), default=0), "best_elo": max((h.elo_rating for h in in_tourn), default=None)}`.
- `is_terminal(memory, cfg) -> bool`: stats = summary_stats; `True` if `stats["num_ideas"] >= cfg.budget.max_ideas`; OR (`in_tourn` non-empty AND `stats["min_matches_per_idea"] >= cfg.budget.max_matches_per_idea`); OR (`cfg.budget.max_tokens` and `memory.tokens_spent >= cfg.budget.max_tokens`); else `False`.
- `decide_next_steps(memory, cfg) -> list[Task]`: track best-Elo history on the instance (`self._elo_hist: list[float]`). Append `summary_stats(memory)["best_elo"]` (skip if None). `in_tourn` = active w/ elo. tasks=[]; if `len(in_tourn) >= 2`: append `Task(RANKING, RUN_TOURNAMENT_BATCH)`. Plateau (OURS): `plateaued = len(self._elo_hist) >= cfg.plateau.window and (self._elo_hist[-1] - self._elo_hist[-cfg.plateau.window]) <= cfg.plateau.epsilon`; if plateaued and `len(in_tourn) >= 2`: append `Task(EVOLUTION, EVOLVE_TOP)`. If `len(in_tourn) >= 2` and `memory.tick % 5 == 0`: append `Task(PROXIMITY, UPDATE_PROXIMITY)` and `Task(META_REVIEW, GENERATE_SYSTEM_FEEDBACK)`. Return tasks.

- [ ] **Step 1: Write the failing test** — `tests/test_supervisor.py`:
```python
import pytest
from cosci.supervisor import Supervisor
from cosci.memory import ContextMemory
from cosci.models import (Hypothesis, ResearchPlan, Task, AgentName, TaskType, Safety)
from cosci.agents.base import Results
from cosci.config import load_config
from tests.fake_llm import FakeLLM

def _sup():
    return Supervisor(agents={})  # route not exercised in these unit tests

@pytest.mark.asyncio
async def test_start_unsafe_goal_aborts():
    cfg = load_config("config.yaml"); mem = ContextMemory()
    llm = FakeLLM(lambda a, m: "this is dangerous\nsafety: unsafe")
    assert await _sup().start("make a bioweapon", mem, llm, cfg) is None

@pytest.mark.asyncio
async def test_start_safe_goal_parses_plan_and_seeds_generation():
    cfg = load_config("config.yaml"); mem = ContextMemory()
    def router(a, m):
        c = m[-1]["content"]
        if "safety" in c.lower() and "research goal" in c.lower(): return "safety: safe"
        return '{"goal": "cure X", "preferences": ["novel"], "attributes": [], "constraints": ["ethical"]}'
    task = await _sup().start("cure X", mem, FakeLLM(router), cfg)
    assert task.agent == AgentName.GENERATION and task.action == TaskType.CREATE_INITIAL_HYPOTHESES
    assert mem.research_plan.goal == "cure X" and mem.research_plan.preferences == ["novel"]

def test_manage_follow_ups_quarantines_unsafe():
    mem = ContextMemory()
    safe = Hypothesis(id="G1", text="t", title="T", source_strategy="s", safety=Safety.SAFE)
    unsafe = Hypothesis(id="G2", text="t", title="T", source_strategy="s", safety=Safety.UNSAFE)
    mem.add_hypothesis(safe); mem.add_hypothesis(unsafe)
    res = Results(follow_ups=[
        Task(agent=AgentName.RANKING, action=TaskType.ADD_TO_TOURNAMENT, target_id="G1"),
        Task(agent=AgentName.RANKING, action=TaskType.ADD_TO_TOURNAMENT, target_id="G2"),
    ])
    kept = _sup().manage_follow_ups(res, mem)
    assert [t.target_id for t in kept] == ["G1"]   # G2 (unsafe) dropped

def test_is_terminal_on_max_ideas():
    cfg = load_config("config.yaml"); mem = ContextMemory()
    for i in range(cfg.budget.max_ideas):
        mem.add_hypothesis(Hypothesis(id=f"G{i}", text="t", title="T", source_strategy="s"))
    assert _sup().is_terminal(mem, cfg) is True

def test_decide_next_steps_emits_tournament_when_two_in_tournament():
    cfg = load_config("config.yaml"); mem = ContextMemory()
    for hid in ("G1", "G2"):
        mem.add_hypothesis(Hypothesis(id=hid, text="t", title="T", source_strategy="s", elo_rating=1200))
    tasks = _sup().decide_next_steps(mem, cfg)
    assert any(t.action == TaskType.RUN_TOURNAMENT_BATCH for t in tasks)
```

- [ ] Steps 2–4 (FAIL → implement `cosci/supervisor.py` → PASS). **Step 5: Commit** `feat(orchestration): Supervisor (chaining, safety quarantine, termination, idle policy)`.

---

## Task 2: Engine (the run loop)

**Files:** Create `cosci/engine.py`; Test `tests/test_engine.py`

**Interfaces / behavior:** `async run_engine(goal_raw, memory, llm, cfg, agents, mode="continuous", snapshot_path=None) -> str | None`.
- `sup = Supervisor(agents)`; `initial = await sup.start(goal_raw, memory, llm, cfg)`. If `initial is None`: return `None` (unsafe-goal abort, no overview).
- `lock = asyncio.Lock()` (guards memory; OURS — sequential loop so it's a correctness belt-and-suspenders for future concurrency).
- **continuous mode:** `queue = GlobalTaskQueue()`; `await queue.put(initial)`; `idle = 0`; loop `while not sup.is_terminal(memory, cfg):` — if `queue.empty()`: `nxt = sup.decide_next_steps(memory, cfg)`; if not nxt → `idle += 1; if idle > 3: break; continue`; else `idle = 0; for t in nxt: await queue.put(t); continue`. Else: `t = await queue.get()`; `async with lock:` `agent = sup.route(t)`; `res = await agent.execute(t, memory, llm, cfg)`; `memory.tick += 1`; `for ft in sup.manage_follow_ups(res, memory): await queue.put(ft)`; if `snapshot_path and memory.tick % 10 == 0: memory.save_snapshot(snapshot_path)`.
- **round_based mode:** repeat up to `cfg.budget.max_ideas` rounds or until `is_terminal`: run a fixed phase sequence by directly routing one task of each: CREATE_INITIAL_HYPOTHESES (first round only) → drain resulting REVIEW follow-ups → ADD_TO_TOURNAMENT for each reviewed → RUN_TOURNAMENT_BATCH → (every other round) EVOLVE_TOP → UPDATE_PROXIMITY. Simpler/deterministic; share `manage_follow_ups` for chaining. (Keep it minimal — the continuous mode is the primary engine; round_based is the cheap deterministic alternative.)
- **Final overview (both modes):** `ov = await agents[AgentName.META_REVIEW].execute(Task(agent=AgentName.META_REVIEW, action=TaskType.GENERATE_FINAL_OVERVIEW), memory, llm, cfg)`; return `ov.overview`.

- [ ] **Step 1: Write the failing test** — `tests/test_engine.py`. A FakeLLM router covering every agent, and a tiny budget so the run terminates fast:
```python
import pytest
from cosci.engine import run_engine
from cosci.memory import ContextMemory
from cosci.config import load_config
from cosci.models import AgentName
from cosci.agents import (GenerationAgent, ReflectionAgent, RankingAgent,
                          EvolutionAgent, ProximityAgent, MetaReviewAgent)
from tests.fake_llm import FakeLLM

class _Enc:  # proximity fake encoder
    def encode(self, texts):
        return [[float(len(t)), 1.0] for t in texts]

def _agents():
    return {AgentName.GENERATION: GenerationAgent(strategies=["literature_review"]),
            AgentName.REFLECTION: ReflectionAgent(),
            AgentName.RANKING: RankingAgent(),
            AgentName.EVOLUTION: EvolutionAgent(),
            AgentName.PROXIMITY: ProximityAgent(encoder=_Enc()),
            AgentName.META_REVIEW: MetaReviewAgent()}

def _router(agent, messages):
    c = messages[-1]["content"].lower()
    if "safety" in c and "research goal" in c: return "safety: safe"
    if "structured research plan" in c or "preferences, attributes" in c:
        return '{"goal": "cure X", "preferences": ["novel"], "attributes": [], "constraints": []}'
    if "deep verification" in c: return "verification: verified"
    if "better idea" in c or "comparative analysis" in c or "better hypothesis" in c:
        return "debate\nbetter idea: 1"
    if "meta-analysis" in c or "research overview" in c or "research directions" in c:
        return "Overview: direction A."
    if "safety:" in c or "rapid initial" in c or "literature-grounded review" in c:
        return "looks fine\nsafety: safe"
    return "Hypothesis: mechanism M explains the effect in detail."

@pytest.mark.asyncio
async def test_engine_runs_to_completion_and_returns_overview(tmp_path):
    cfg = load_config("config.yaml")
    cfg.budget.max_ideas = 2; cfg.budget.max_matches_per_idea = 1   # tiny budget
    mem = ContextMemory()
    overview = await run_engine("cure X", mem, FakeLLM(_router), cfg, _agents(), mode="continuous")
    assert isinstance(overview, str) and len(overview) > 0
    assert len(mem.active_hypotheses()) >= 1          # generated hypotheses exist
    assert any(r for rs in mem.reviews.values() for r in rs)  # reviews happened
    # (a tournament match may or may not occur depending on timing; overview is the key contract)

@pytest.mark.asyncio
async def test_engine_aborts_on_unsafe_goal():
    cfg = load_config("config.yaml"); mem = ContextMemory()
    llm = FakeLLM(lambda a, m: "dangerous\nsafety: unsafe")
    assert await run_engine("bioweapon", mem, llm, cfg, _agents()) is None
```

- [ ] Steps 2–4 (FAIL → implement `cosci/engine.py` → PASS; ensure NO infinite loop — the `idle > 3` break and `is_terminal` guarantee termination). **Step 5: Run FULL suite from repo root**, then **Commit** `feat(orchestration): queue-driven engine (continuous + round_based)`.

---

## Phase 4 done-criteria
- Full suite green from repo root; a tiny end-to-end run completes, produces an overview, and respects the unsafe-goal abort + unsafe-hypothesis quarantine.
- `Supervisor` + `run_engine` exposed; safety quarantine enforced in `manage_follow_ups`; termination honors SN8 guard + budget.

## Self-review notes
- **Spec coverage:** spec §4 (data flow / chaining / termination), §10 (test-time scaling via budget), §12 (safety gates) → Tasks 1–2; the quarantine (§4 step 3) is enforced in `manage_follow_ups`.
- **Deferred (documented):** true concurrent worker dispatch (Methods' async framework) — the loop is sequential per SN8; `cfg.workers`/the lock are placeholders for a future compute/commit-split enhancement. Real grounding still Phase 5; CLI still Phase 6.
- **Termination safety:** the loop cannot hang — `is_terminal` (budget) plus the `idle > 3` empty-queue break bound it.
- **Type consistency:** `Supervisor(agents: dict[AgentName, Agent])`, `run_engine(...)`, `manage_follow_ups`/`decide_next_steps`/`is_terminal` names reused; agents/Results/TaskType/Safety from Phases 1–3.
