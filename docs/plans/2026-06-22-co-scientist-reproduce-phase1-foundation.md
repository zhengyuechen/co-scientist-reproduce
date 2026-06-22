# Co-Scientist Reproduction — Phase 1 (Foundation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the foundation layer (config, domain models, Elo math, task queue, LLM client + FakeLLM, persistent context memory) that every later phase depends on.

**Architecture:** Pure, side-effect-light modules under `cosci/`, each one responsibility, each unit-tested against a deterministic `FakeLLM` (no API calls). The async engine, agents, tools, and CLI are built on top of these in Phases 2–6.

**Tech Stack:** Python 3.11 (conda env `cosci-reproduce`), `pydantic` v2 (models + snapshot serialization), `pyyaml` + `python-dotenv` (config), `tenacity` (LLM retry), `openai` SDK (OpenRouter), `pytest` + `pytest-asyncio`.

**Plan series:** This is **Plan 1 of 6**. Foundation is fully detailed here; Phases 2–6 are roadmapped at the end (§ Roadmap) and each becomes its own detailed plan when reached.

## Global Constraints

- Python **3.11**, conda env **`cosci-reproduce`**. All test/run commands use `conda run -n cosci-reproduce …`.
- Project root: `/Users/jeremychen/My Drive/Ran's Lab/projects/research-agents/co-scientist-reproduce/`. All paths below are relative to it.
- LLM backend is **OpenRouter** via the `openai` SDK with `base_url="https://openrouter.ai/api/v1"`; model is **per-agent configurable**. Never hardcode a key — read `OPENROUTER_API_KEY` from `.env`/env.
- **Provenance discipline:** every constant/prompt is tagged in code as `SM` (from supplement), `SM-pseudocode`, or `OURS`/`RECONSTRUCTED`. Foundation constants are all `OURS`.
- Elo: initial rating **1200** (SM), update **OURS** K=32 / scale=400.
- TDD: write the failing test first, watch it fail, minimal impl, watch it pass, commit. One logical change per commit.
- DRY, YAGNI: build only what the spec needs; no speculative abstraction.

---

## File structure (Phase 1)

| File | Responsibility |
|---|---|
| `requirements.txt` | Pinned-enough dependency list |
| `config.yaml` | Default run configuration (models, budgets, thresholds) |
| `.env.example` | `OPENROUTER_API_KEY`, optional `WEB_SEARCH_API_KEY` |
| `.gitignore` | `.env`, `results/`, `snapshots/`, `__pycache__/` |
| `cosci/__init__.py` | Package marker |
| `cosci/config.py` | Load+validate config.yaml + .env → typed `Config` |
| `cosci/models.py` | `Hypothesis`, `Review`, `ResearchPlan`, `Task`, `MatchResult`, enums |
| `cosci/elo.py` | Pure Elo expected-score + update |
| `cosci/tasks.py` | `GlobalTaskQueue` (async priority queue over `Task`) |
| `cosci/llm.py` | `LLMClient` protocol, `OpenRouterClient`, `extract_json` |
| `cosci/memory.py` | `ContextMemory` + JSON snapshot save/load |
| `tests/fake_llm.py` | Deterministic `FakeLLM` for all later tests |
| `tests/test_*.py` | One test module per unit above |

---

## Task 1: Project scaffold + config

**Files:**
- Create: `requirements.txt`, `config.yaml`, `.env.example`, `.gitignore`, `cosci/__init__.py`, `cosci/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `cosci.config.load_config(path: str = "config.yaml") -> Config`. `Config` (pydantic) fields: `models: dict[str,str]`, `default_model: str`, `elo: EloCfg(k_factor:int, scale:int)`, `budget: BudgetCfg(max_ideas:int, max_matches_per_idea:int, max_wallclock_s:int, max_tokens:int|None)`, `temperature: dict[str,float]`, `debate: DebateCfg(turns_typical_min:int, turns_typical_max:int, turns_max:int)`, `evolution: EvoCfg(top_k:int)`, `overview: OverviewCfg(top_n:int)`, `workers:int`, `scheduler:str`, `proximity: ProxCfg(method:str, model:str)`, `grounding: GroundCfg(backend:str)`, `plateau: PlateauCfg(window:int, epsilon:float)`. `Config.model_for(agent:str) -> str` returns `models.get(agent, default_model)`.

- [ ] **Step 1: Create the conda env and dependency file**

Create `requirements.txt`:
```
openai>=1.0
pydantic>=2.0
pyyaml>=6.0
python-dotenv>=1.0
tenacity>=8.0
sentence-transformers>=2.2
arxiv>=2.0
pytest>=8.0
pytest-asyncio>=0.23
```

Run:
```bash
conda create -n cosci-reproduce python=3.11 -y
conda run -n cosci-reproduce pip install -r requirements.txt
```
Expected: env created; install ends with `Successfully installed …`.

- [ ] **Step 2: Create config.yaml, .env.example, .gitignore**

`config.yaml`:
```yaml
# Provenance: SM = from supplement, OURS = our default (see spec §9)
default_model: "google/gemini-2.0-flash-exp:free"   # OURS (free, cost-safe)
models: {}                                           # per-agent overrides, e.g. generation: "..."
elo:        { k_factor: 32, scale: 400 }             # OURS (SM gives init=1200 only)
budget:     { max_ideas: 20, max_matches_per_idea: 8, max_wallclock_s: 1800, max_tokens: null }  # OURS
temperature: { generation: 0.7, reflection: 0.5, ranking: 0.3, evolution: 0.8 }  # OURS
debate:     { turns_typical_min: 3, turns_typical_max: 5, turns_max: 10 }  # SM (SN9)
evolution:  { top_k: 5 }                              # SM (SN8)
overview:   { top_n: 10 }                             # SM (SN8)
workers: 4                                            # SM Fig 1b (illustrative)
scheduler: "continuous"                               # OURS flag: continuous | round_based
proximity:  { method: "embeddings", model: "sentence-transformers/all-MiniLM-L6-v2" }
grounding:  { backend: "arxiv" }                      # OURS: arxiv | tavily | serper | brave
plateau:    { window: 10, epsilon: 5.0 }              # OURS (best-Elo Δ over window)
```

`.env.example`:
```
OPENROUTER_API_KEY=your_key_here
# Optional, only if grounding.backend is a web backend:
WEB_SEARCH_API_KEY=
```

`.gitignore`:
```
.env
results/
snapshots/
__pycache__/
*.pyc
.pytest_cache/
```

- [ ] **Step 3: Write the failing test**

`tests/test_config.py`:
```python
from cosci.config import load_config

def test_load_defaults(tmp_path):
    cfg_text = (
        'default_model: "m/free"\n'
        'models: { generation: "m/strong" }\n'
        'elo: { k_factor: 32, scale: 400 }\n'
        'budget: { max_ideas: 20, max_matches_per_idea: 8, max_wallclock_s: 1800, max_tokens: null }\n'
        'temperature: { generation: 0.7, reflection: 0.5, ranking: 0.3, evolution: 0.8 }\n'
        'debate: { turns_typical_min: 3, turns_typical_max: 5, turns_max: 10 }\n'
        'evolution: { top_k: 5 }\n'
        'overview: { top_n: 10 }\n'
        'workers: 4\n'
        'scheduler: "continuous"\n'
        'proximity: { method: "embeddings", model: "mini" }\n'
        'grounding: { backend: "arxiv" }\n'
        'plateau: { window: 10, epsilon: 5.0 }\n'
    )
    p = tmp_path / "config.yaml"
    p.write_text(cfg_text)
    cfg = load_config(str(p))
    assert cfg.elo.k_factor == 32
    assert cfg.budget.max_ideas == 20
    assert cfg.model_for("generation") == "m/strong"   # override used
    assert cfg.model_for("ranking") == "m/free"          # falls back to default
    assert cfg.scheduler == "continuous"
```

- [ ] **Step 4: Run test to verify it fails**

Run: `conda run -n cosci-reproduce pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cosci.config'`.

- [ ] **Step 5: Write minimal implementation**

`cosci/__init__.py`: (empty file)

`cosci/config.py`:
```python
"""Typed configuration loaded from config.yaml + .env. All defaults are OURS (spec §9)."""
from __future__ import annotations
import os
import yaml
from pydantic import BaseModel
from dotenv import load_dotenv

class EloCfg(BaseModel):
    k_factor: int = 32
    scale: int = 400

class BudgetCfg(BaseModel):
    max_ideas: int = 20
    max_matches_per_idea: int = 8
    max_wallclock_s: int = 1800
    max_tokens: int | None = None

class DebateCfg(BaseModel):
    turns_typical_min: int = 3
    turns_typical_max: int = 5
    turns_max: int = 10

class EvoCfg(BaseModel):
    top_k: int = 5

class OverviewCfg(BaseModel):
    top_n: int = 10

class ProxCfg(BaseModel):
    method: str = "embeddings"
    model: str = "sentence-transformers/all-MiniLM-L6-v2"

class GroundCfg(BaseModel):
    backend: str = "arxiv"

class PlateauCfg(BaseModel):
    window: int = 10
    epsilon: float = 5.0

class Config(BaseModel):
    default_model: str
    models: dict[str, str] = {}
    elo: EloCfg = EloCfg()
    budget: BudgetCfg = BudgetCfg()
    temperature: dict[str, float] = {}
    debate: DebateCfg = DebateCfg()
    evolution: EvoCfg = EvoCfg()
    overview: OverviewCfg = OverviewCfg()
    workers: int = 4
    scheduler: str = "continuous"
    proximity: ProxCfg = ProxCfg()
    grounding: GroundCfg = GroundCfg()
    plateau: PlateauCfg = PlateauCfg()

    def model_for(self, agent: str) -> str:
        return self.models.get(agent, self.default_model)

def load_config(path: str = "config.yaml") -> Config:
    load_dotenv()  # populate os.environ from .env if present
    with open(path) as f:
        data = yaml.safe_load(f)
    return Config(**data)

def require_openrouter_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set (add it to .env or the environment).")
    return key
```

- [ ] **Step 6: Run test to verify it passes**

Run: `conda run -n cosci-reproduce pytest tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add requirements.txt config.yaml .env.example .gitignore cosci/__init__.py cosci/config.py tests/test_config.py
git commit -m "feat(foundation): typed config loader + project scaffold"
```
(If the repo isn't initialized: `git init` first, or skip commits if running without git — note in the run log.)

---

## Task 2: Domain models

**Files:**
- Create: `cosci/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces:
  - Enums (str-valued): `AgentName{GENERATION,REFLECTION,RANKING,EVOLUTION,PROXIMITY,META_REVIEW}`, `TaskType{CREATE_INITIAL_HYPOTHESES,REVIEW_HYPOTHESIS,ADD_TO_TOURNAMENT,RUN_TOURNAMENT_BATCH,EVOLVE_TOP,UPDATE_PROXIMITY,GENERATE_SYSTEM_FEEDBACK,GENERATE_FINAL_OVERVIEW}`, `Safety{UNREVIEWED,SAFE,UNSAFE}`, `DebateMode{SINGLE_TURN,MULTI_TURN}`.
  - `ResearchPlan(goal:str, preferences:list[str], attributes:list[str], constraints:list[str])`.
  - `Hypothesis(id:str, text:str, title:str, source_strategy:str, parent_ids:list[str]=[], elo_rating:float|None=None, matches_played:int=0, active:bool=True, created_tick:int=0, safety:Safety=UNREVIEWED, safety_reason:str|None=None, origin:str="generated")`.
  - `Review(hypothesis_id:str, type:str, scores:dict[str,float]={}, text:str="", references:list[str]=[], tool_grounded:bool=False, safety:Safety=UNREVIEWED, safety_reason:str|None=None)`.
  - `Task(agent:AgentName, action:TaskType, target_id:str|None=None, priority:int=5, payload:dict={})`.
  - `MatchResult(a_id:str, b_id:str, mode:DebateMode, winner_id:str, loser_id:str, elo_before:dict[str,float], elo_after:dict[str,float], transcript:str, tick:int)`.

- [ ] **Step 1: Write the failing test**

`tests/test_models.py`:
```python
from cosci.models import Hypothesis, Safety, Task, AgentName, TaskType

def test_hypothesis_defaults():
    h = Hypothesis(id="G1", text="t", title="T", source_strategy="literature_review")
    assert h.elo_rating is None
    assert h.active is True
    assert h.safety == Safety.UNREVIEWED
    assert h.origin == "generated"

def test_task_roundtrip_json():
    t = Task(agent=AgentName.REFLECTION, action=TaskType.REVIEW_HYPOTHESIS, target_id="G1")
    dumped = t.model_dump_json()
    assert '"target_id":"G1"' in dumped.replace(" ", "")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n cosci-reproduce pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cosci.models'`.

- [ ] **Step 3: Write minimal implementation**

`cosci/models.py`:
```python
"""Domain models. Field set matches spec §8 context-memory schema."""
from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field

class AgentName(str, Enum):
    GENERATION = "generation"
    REFLECTION = "reflection"
    RANKING = "ranking"
    EVOLUTION = "evolution"
    PROXIMITY = "proximity"
    META_REVIEW = "meta_review"

class TaskType(str, Enum):
    CREATE_INITIAL_HYPOTHESES = "create_initial_hypotheses"
    REVIEW_HYPOTHESIS = "review_hypothesis"
    ADD_TO_TOURNAMENT = "add_to_tournament"
    RUN_TOURNAMENT_BATCH = "run_tournament_batch"
    EVOLVE_TOP = "evolve_top"
    UPDATE_PROXIMITY = "update_proximity"
    GENERATE_SYSTEM_FEEDBACK = "generate_system_feedback"
    GENERATE_FINAL_OVERVIEW = "generate_final_overview"

class Safety(str, Enum):
    UNREVIEWED = "unreviewed"
    SAFE = "safe"
    UNSAFE = "unsafe"

class DebateMode(str, Enum):
    SINGLE_TURN = "single_turn"
    MULTI_TURN = "multi_turn"

class ResearchPlan(BaseModel):
    goal: str
    preferences: list[str] = []
    attributes: list[str] = []
    constraints: list[str] = []

class Hypothesis(BaseModel):
    id: str
    text: str
    title: str
    source_strategy: str
    parent_ids: list[str] = []
    elo_rating: float | None = None
    matches_played: int = 0
    active: bool = True
    created_tick: int = 0
    safety: Safety = Safety.UNREVIEWED
    safety_reason: str | None = None
    origin: str = "generated"  # generated | evolved | user_seed

class Review(BaseModel):
    hypothesis_id: str
    type: str
    scores: dict[str, float] = {}
    text: str = ""
    references: list[str] = []
    tool_grounded: bool = False
    safety: Safety = Safety.UNREVIEWED
    safety_reason: str | None = None

class Task(BaseModel):
    agent: AgentName
    action: TaskType
    target_id: str | None = None
    priority: int = 5  # lower = sooner
    payload: dict = Field(default_factory=dict)

class MatchResult(BaseModel):
    a_id: str
    b_id: str
    mode: DebateMode
    winner_id: str
    loser_id: str
    elo_before: dict[str, float]
    elo_after: dict[str, float]
    transcript: str
    tick: int
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n cosci-reproduce pytest tests/test_models.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cosci/models.py tests/test_models.py
git commit -m "feat(foundation): domain models + enums"
```

---

## Task 3: Elo math

**Files:**
- Create: `cosci/elo.py`
- Test: `tests/test_elo.py`

**Interfaces:**
- Produces: `expected_score(r_a:float, r_b:float, scale:int=400) -> float`; `update(winner:float, loser:float, k:int=32, scale:int=400) -> tuple[float,float]` returning `(new_winner, new_loser)`.

- [ ] **Step 1: Write the failing test**

`tests/test_elo.py`:
```python
from cosci.elo import expected_score, update

def test_expected_score_equal_ratings():
    assert abs(expected_score(1200, 1200) - 0.5) < 1e-9

def test_update_equal_ratings_k32():
    w, l = update(1200, 1200, k=32, scale=400)
    assert abs(w - 1216.0) < 1e-6   # 1200 + 32*(1-0.5)
    assert abs(l - 1184.0) < 1e-6   # 1200 + 32*(0-0.5)

def test_update_is_zero_sum():
    w, l = update(1300, 1100, k=32)
    assert abs((w - 1300) + (l - 1100)) < 1e-9  # points conserved
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n cosci-reproduce pytest tests/test_elo.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cosci.elo'`.

- [ ] **Step 3: Write minimal implementation**

`cosci/elo.py`:
```python
"""Elo ranking. OURS: K=32, scale=400 (SM specifies only initial rating 1200)."""

def expected_score(r_a: float, r_b: float, scale: int = 400) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / scale))

def update(winner: float, loser: float, k: int = 32, scale: int = 400) -> tuple[float, float]:
    e_w = expected_score(winner, loser, scale)
    e_l = 1.0 - e_w
    new_winner = winner + k * (1.0 - e_w)
    new_loser = loser + k * (0.0 - e_l)
    return new_winner, new_loser
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n cosci-reproduce pytest tests/test_elo.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cosci/elo.py tests/test_elo.py
git commit -m "feat(foundation): pure Elo expected-score + update"
```

---

## Task 4: Task queue

**Files:**
- Create: `cosci/tasks.py`
- Test: `tests/test_tasks.py`

**Interfaces:**
- Produces: `GlobalTaskQueue` with async methods `put(task:Task)`, `get() -> Task`, and sync `empty() -> bool`, `qsize() -> int`. Ordering: by `task.priority` ascending (lower sooner), FIFO within equal priority. Internally wraps `asyncio.PriorityQueue` with a monotonic counter so `Task` objects are never compared.

- [ ] **Step 1: Write the failing test**

`tests/test_tasks.py`:
```python
import asyncio
import pytest
from cosci.models import Task, AgentName, TaskType
from cosci.tasks import GlobalTaskQueue

def _t(priority, target):
    return Task(agent=AgentName.RANKING, action=TaskType.RUN_TOURNAMENT_BATCH,
                target_id=target, priority=priority)

@pytest.mark.asyncio
async def test_priority_then_fifo():
    q = GlobalTaskQueue()
    await q.put(_t(5, "a"))
    await q.put(_t(1, "b"))   # higher priority (lower number)
    await q.put(_t(5, "c"))
    assert (await q.get()).target_id == "b"  # priority 1 first
    assert (await q.get()).target_id == "a"  # then FIFO among priority 5
    assert (await q.get()).target_id == "c"
    assert q.empty() is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n cosci-reproduce pytest tests/test_tasks.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cosci.tasks'`.

- [ ] **Step 3: Write minimal implementation**

`cosci/tasks.py`:
```python
"""Async priority queue over Task. Lower priority value = scheduled sooner; FIFO within a level."""
from __future__ import annotations
import asyncio
import itertools
from cosci.models import Task

class GlobalTaskQueue:
    def __init__(self) -> None:
        self._q: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._counter = itertools.count()

    async def put(self, task: Task) -> None:
        # (priority, insertion_order) makes ordering total without comparing Task objects
        await self._q.put((task.priority, next(self._counter), task))

    async def get(self) -> Task:
        _, _, task = await self._q.get()
        return task

    def empty(self) -> bool:
        return self._q.empty()

    def qsize(self) -> int:
        return self._q.qsize()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n cosci-reproduce pytest tests/test_tasks.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cosci/tasks.py tests/test_tasks.py
git commit -m "feat(foundation): async priority task queue"
```

---

## Task 5: LLM client + FakeLLM + JSON extraction

**Files:**
- Create: `cosci/llm.py`, `tests/fake_llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Produces:
  - `extract_json(text:str) -> object` — strips ```` ```json ```` / ```` ``` ```` fences and parses; raises `ValueError` on failure.
  - `LLMClient` (typing.Protocol): `complete(agent:str, messages:list[dict], temperature:float|None=None, max_tokens:int|None=None) -> str`.
  - `OpenRouterClient(config:Config)` implementing `LLMClient` (tenacity retry; uses `config.model_for(agent)` and `config.temperature.get(agent)`).
  - `FakeLLM(router:Callable[[str,list[dict]],str])` in `tests/fake_llm.py` implementing `LLMClient` deterministically; records `.calls`.

- [ ] **Step 1: Write the failing test**

`tests/test_llm.py`:
```python
import pytest
from cosci.llm import extract_json
from tests.fake_llm import FakeLLM

def test_extract_json_strips_fences():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('{"b": 2}') == {"b": 2}

def test_extract_json_raises_on_garbage():
    with pytest.raises(ValueError):
        extract_json("not json at all")

@pytest.mark.asyncio
async def test_fake_llm_is_deterministic_and_records():
    fake = FakeLLM(router=lambda agent, messages: f"reply-to-{agent}")
    out = await fake.complete("generation", [{"role": "user", "content": "hi"}])
    assert out == "reply-to-generation"
    assert fake.calls[0]["agent"] == "generation"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n cosci-reproduce pytest tests/test_llm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cosci.llm'`.

- [ ] **Step 3: Write minimal implementation**

`cosci/llm.py`:
```python
"""LLM access. Only module that talks to OpenRouter. Swappable with FakeLLM in tests."""
from __future__ import annotations
import json
from typing import Protocol, Callable
from tenacity import retry, wait_exponential, stop_after_attempt
from cosci.config import Config, require_openrouter_key

def extract_json(text: str) -> object:
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.startswith("json"):
            s = s[4:]
    if s.endswith("```"):
        s = s[: s.rfind("```")]
    s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"could not parse JSON: {e}") from e

class LLMClient(Protocol):
    async def complete(self, agent: str, messages: list[dict],
                       temperature: float | None = None,
                       max_tokens: int | None = None) -> str: ...

class OpenRouterClient:
    def __init__(self, config: Config) -> None:
        from openai import AsyncOpenAI
        self._cfg = config
        self._client = AsyncOpenAI(
            api_key=require_openrouter_key(),
            base_url="https://openrouter.ai/api/v1",
        )

    @retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(4), reraise=True)
    async def complete(self, agent: str, messages: list[dict],
                       temperature: float | None = None,
                       max_tokens: int | None = None) -> str:
        if temperature is None:
            temperature = self._cfg.temperature.get(agent, 0.7)
        resp = await self._client.chat.completions.create(
            model=self._cfg.model_for(agent),
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""
```

`tests/fake_llm.py`:
```python
"""Deterministic LLM stand-in for unit tests (no network)."""
from __future__ import annotations
from typing import Callable

class FakeLLM:
    def __init__(self, router: Callable[[str, list[dict]], str]) -> None:
        self._router = router
        self.calls: list[dict] = []

    async def complete(self, agent: str, messages: list[dict],
                       temperature: float | None = None,
                       max_tokens: int | None = None) -> str:
        self.calls.append({"agent": agent, "messages": messages, "temperature": temperature})
        return self._router(agent, messages)
```

Also create empty `tests/__init__.py` so `tests.fake_llm` imports cleanly.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n cosci-reproduce pytest tests/test_llm.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cosci/llm.py tests/fake_llm.py tests/__init__.py tests/test_llm.py
git commit -m "feat(foundation): LLM client protocol, OpenRouter impl, FakeLLM, JSON extractor"
```

---

## Task 6: Context memory + JSON snapshot

**Files:**
- Create: `cosci/memory.py`
- Test: `tests/test_memory.py`

**Interfaces:**
- Produces: `ContextMemory` (pydantic) fields: `research_plan:ResearchPlan|None`, `hypotheses:dict[str,Hypothesis]`, `reviews:dict[str,list[Review]]`, `tournament:list[MatchResult]`, `proximity:dict[str,list[dict]]`, `system_feedback:str`, `tick:int`, `tokens_spent:int`, `_id_counter:int`. Methods: `new_id(prefix:str)->str`, `add_hypothesis(h)`, `get(id)->Hypothesis|None`, `active_hypotheses()->list[Hypothesis]` (active and not unsafe), `add_review(r)`, `record_match(m)`, `save_snapshot(path:str)`, classmethod `load_snapshot(path:str)->ContextMemory`.

- [ ] **Step 1: Write the failing test**

`tests/test_memory.py`:
```python
from cosci.memory import ContextMemory
from cosci.models import Hypothesis, Safety, ResearchPlan

def _h(hid, safety=Safety.SAFE, active=True):
    return Hypothesis(id=hid, text="t", title="T", source_strategy="s",
                      safety=safety, active=active)

def test_active_excludes_unsafe_and_inactive():
    m = ContextMemory(research_plan=ResearchPlan(goal="g"))
    m.add_hypothesis(_h("G1"))
    m.add_hypothesis(_h("G2", safety=Safety.UNSAFE))     # quarantined
    m.add_hypothesis(_h("G3", active=False))             # inactive
    ids = {h.id for h in m.active_hypotheses()}
    assert ids == {"G1"}

def test_snapshot_roundtrip(tmp_path):
    m = ContextMemory(research_plan=ResearchPlan(goal="g"))
    m.add_hypothesis(_h("G1"))
    p = tmp_path / "snap.json"
    m.save_snapshot(str(p))
    m2 = ContextMemory.load_snapshot(str(p))
    assert m2.get("G1").title == "T"
    assert m2.research_plan.goal == "g"

def test_new_id_is_unique():
    m = ContextMemory()
    a, b = m.new_id("G"), m.new_id("G")
    assert a != b and a.startswith("G")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n cosci-reproduce pytest tests/test_memory.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cosci.memory'`.

- [ ] **Step 3: Write minimal implementation**

`cosci/memory.py`:
```python
"""ContextMemory == SM 'SharedMemory'. JSON-snapshot for restartability (spec §8).
Concurrency note: callers serialize mutations via an asyncio.Lock in the scheduler (Phase 4);
these methods are plain sync."""
from __future__ import annotations
from pydantic import BaseModel
from cosci.models import Hypothesis, Review, MatchResult, ResearchPlan, Safety

class ContextMemory(BaseModel):
    research_plan: ResearchPlan | None = None
    hypotheses: dict[str, Hypothesis] = {}
    reviews: dict[str, list[Review]] = {}
    tournament: list[MatchResult] = []
    proximity: dict[str, list[dict]] = {}
    system_feedback: str = ""
    tick: int = 0
    tokens_spent: int = 0
    id_counter: int = 0

    def new_id(self, prefix: str) -> str:
        self.id_counter += 1
        return f"{prefix}{self.id_counter}"

    def add_hypothesis(self, h: Hypothesis) -> None:
        self.hypotheses[h.id] = h

    def get(self, hid: str) -> Hypothesis | None:
        return self.hypotheses.get(hid)

    def active_hypotheses(self) -> list[Hypothesis]:
        return [h for h in self.hypotheses.values()
                if h.active and h.safety != Safety.UNSAFE]

    def add_review(self, r: Review) -> None:
        self.reviews.setdefault(r.hypothesis_id, []).append(r)

    def record_match(self, m: MatchResult) -> None:
        self.tournament.append(m)

    def save_snapshot(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load_snapshot(cls, path: str) -> "ContextMemory":
        with open(path) as f:
            return cls.model_validate_json(f.read())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n cosci-reproduce pytest tests/test_memory.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full Phase 1 suite and commit**

Run: `conda run -n cosci-reproduce pytest -v`
Expected: all tests PASS (config, models, elo, tasks, llm, memory).

```bash
git add cosci/memory.py tests/test_memory.py
git commit -m "feat(foundation): context memory + JSON snapshot round-trip"
```

---

## Phase 1 done-criteria

- `conda run -n cosci-reproduce pytest -v` is green.
- `cosci/` exposes: `config.load_config`, `models.*`, `elo.expected_score/update`, `tasks.GlobalTaskQueue`, `llm.LLMClient/OpenRouterClient/extract_json`, `memory.ContextMemory`.
- No agent/engine logic yet — that's Phase 2+.

---

## Roadmap — Phases 2–6 (each becomes its own detailed plan when reached)

Each task below lists its deliverable + key interface; full bite-sized TDD steps are written just-in-time. Boundaries and signatures are locked here so the phases compose.

### Phase 2 — Prompts (`prompts/`)
- **2.1 `prompts/verbatim.py`** — the 8 SM-verbatim SN9 prompts (spec Appendix A), byte-faithful, each with `# SN9.x` + a `PLACEHOLDER_MAP`. Test: every documented placeholder present; map covers all verbatim names. *Produces:* named prompt constants + `PLACEHOLDER_MAP`.
- **2.2 `prompts/reconstructed.py`** — in-style prompts for the Appendix C gaps (generation iterative-assumptions/research-expansion; reflection initial/full/deep_verification/simulation/recurrent; evolution combine/simplify/grounding/inspiration; meta overview/system_feedback; supervisor goal_parse/safety_review). Each tagged `# RECONSTRUCTED`. Test: each constant non-empty + has required placeholders.
- **2.3 `prompts/render.py`** — `render(template:str, **vars) -> str` applying `PLACEHOLDER_MAP`, plus `assemble_instructions(memory)` (injects `system_feedback` into `{instructions}`/`{notes}`). Test: all placeholders filled; missing var raises.

### Phase 3 — Agents (`cosci/agents/`)
Each agent: `async execute(task:Task, memory:ContextMemory, llm:LLMClient, cfg:Config) -> Results` where `Results` carries new/updated objects + follow-up `Task` requests. Built and tested against `FakeLLM`.
- **3.1 `agents/base.py`** — `Results` dataclass + `Agent` protocol. *Produces:* `Results(new_hypotheses, reviews, match, follow_ups, overview, feedback)`.
- **3.2 generation.py** — strategies literature_review (SM), scientific_debate/self-play (SM), iterative_assumptions* , research_expansion*; parses N hypotheses; emits `NewHypothesisCreated` follow-ups. Test: FakeLLM returns 2 hypotheses → 2 Hypothesis objs + 2 reflection follow-ups.
- **3.3 reflection.py** — review types initial/full/deep_verification/observation(SM)/simulation/recurrent; emits **safety verdict** (drives quarantine) + `ReviewCompleted`. Test: unsafe verdict → review.safety==UNSAFE.
- **3.4 ranking.py** — `add_to_tournament` (idempotent, Elo=1200), `run_tournament_batch` (match-select via proximity+new+top; single vs multi-turn by rank; parse `better idea|hypothesis: <1|2>`; apply `elo.update`). Test: deterministic FakeLLM debate → correct winner Elo↑.
- **3.5 evolution.py** — combine/simplify/feasibility(SM)/out_of_box(SM)/grounding/inspiration on top-5; evolved → reviewed → tournament. Test: top-5 selection + 1 evolved hypothesis emitted with parent_ids.
- **3.6 proximity.py** — `update_proximity_graph` via sentence-transformers embeddings (lazy-loaded; injectable encoder for tests). Test: identical texts → similarity ≈ 1.0 with a fake encoder.
- **3.7 meta_review.py** — `generate_system_feedback` (SM synthesis) → `memory.system_feedback`; `generate_final_overview` (top-10). Test: feedback string written to memory.

### Phase 4 — Orchestration (`cosci/supervisor.py`, `cosci/scheduler.py`)
- **4.1 supervisor.py** — `start(goal)` (safety goal-gate → parse plan → seed first task), `manage_follow_ups(results)` (chaining incl. **unsafe→quarantine, no AddToTournament**), `decide_next_steps()` (tournament batch; plateau→evolve; ticks→feedback/overview), `summary_stats()`, agent-sampling weights, `is_terminal()` (ideas≥max OR min-matches≥max OR budget). Tests: chaining new→review→tournament; unsafe never enqueued; termination OR-logic; plateau trigger.
- **4.2 scheduler.py** — continuous async worker pool (N workers, `asyncio.Lock` around memory mutations) + `round_based` mode (flag). Integration test with FakeLLM: a full tiny run reaches terminal + produces an overview; per-task failure is isolated; snapshot every K tasks.

### Phase 5 — Tools / grounding (`cosci/tools/`)
- **5.1 tools/web_search.py** — `WebSearchBackend` protocol (`search`, `fetch`); `ArxivBackend` (default, free); optional Tavily/Serper/Brave by key; `faithful-grounding` flag + startup warning when arxiv-only. Test: ArxivBackend parsing with a recorded fixture (no live net).
- **5.2 wire-in** — generation literature_review + reflection full use the backend; graceful parametric fallback with logged note. Test: no-backend path still produces a hypothesis.

### Phase 6 — CLI + logging + results (`cosci/cli.py`, `cosci/logging_utils.py`)
- **6.1 logging_utils.py** — structured console + JSON/markdown writers; `elo_trajectory.csv`. Test: writer emits the spec §14 files.
- **6.2 cli.py** — entrypoint: goal (+ `--seed-hypotheses`, `--constraints`, `--resume`, `--feedback`); builds Config+OpenRouterClient; runs scheduler; writes `results/<ts>_<slug>/`. Test: `--help` and a `--dry-run` that wires FakeLLM end-to-end and writes a results dir.
- **6.3 README + live smoke test** — gated 2-hypothesis run on the free model; README quickstart.

---

## Self-review notes (Phase 1)
- **Spec coverage (Phase 1 scope):** config §9 → Task 1; models/schema §8 → Task 2; Elo §7 → Task 3; queue §3/§4 → Task 4; LLM/FakeLLM §3/§13 → Task 5; context-memory + snapshot §8 → Task 6. Later spec sections are mapped to Phases 2–6 in the roadmap.
- **No placeholders:** every code/test/command step contains real content.
- **Type consistency:** `Config.model_for`, `Task.priority`, `Hypothesis` fields, `MatchResult`, and `ContextMemory.active_hypotheses` names are reused identically by the roadmap interface blocks.
