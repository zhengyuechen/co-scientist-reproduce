# Co-Scientist Reproduction — Phase 5 (Grounding) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Give the agents real literature grounding — a pluggable web-search layer (arXiv default, optional web backend) wired into generation's literature strategy and reflection's full review — and clear the two Phase 4 follow-ups.

**Architecture:** `cosci/tools/web_search.py` defines an `Article`, a `WebSearchBackend` Protocol, a free `ArxivBackend` (injectable client, `arxiv` lazy-imported), an optional `TavilyBackend` (httpx, needs key), a `build_backend(cfg)` factory, and `format_articles`/`is_faithful_grounding` helpers. Generation and Reflection accept an injected `grounding` backend; when present they search → format → fill the `articles_with_reasoning` slot, else fall back to `""` (parametric) — exactly the graceful-degradation design.

**Tech Stack:** Python 3.11 (env `cosci-reproduce`), `arxiv`, `httpx`, `pytest`/`pytest-asyncio`. No live network in tests (injected fakes / monkeypatch).

**Plan series:** Plan 5 of 6. Builds on Phases 1–4 (merged to `main`).

## Global Constraints

- Python 3.11, env `cosci-reproduce`; tests via `conda run -n cosci-reproduce pytest` **from the repo root**.
- Repo `/Users/jeremychen/My Drive/Ran's Lab/projects/research-agents/co-scientist-reproduce`, a git repo. Work on branch `phase-5-grounding` (NOT main).
- **No live network in tests.** `ArxivBackend` takes an injectable client; the `arxiv`/`httpx` imports are lazy (inside methods) so importing the module needs neither installed-at-top nor network.
- **Fidelity:** arXiv-only grounding is a documented compromise (the paper used broad web search). `is_faithful_grounding(cfg)` is True only for a real web backend. Backend protocol is OURS; provenance comment it.
- **Backward compatible:** generation/reflection gain a `grounding=None` param; with `None` they behave exactly as Phase 3 (empty articles). Existing tests stay green.
- Commit messages plain — NO attribution trailer. TDD; DRY; YAGNI.

## File structure (Phase 5)

| File | Responsibility |
|---|---|
| `cosci/tools/__init__.py` | package marker |
| `cosci/tools/web_search.py` | `Article`, `WebSearchBackend`, `ArxivBackend`, `TavilyBackend`, `build_backend`, `format_articles`, `is_faithful_grounding` |
| `cosci/agents/generation.py` (modify) | literature_review uses injected grounding |
| `cosci/agents/reflection.py` (modify) | full review uses injected grounding |
| `cosci/engine.py` (modify) | round_based: chain evolved-hypothesis REVIEW; `run_engine` overview empty-guard |
| `tests/test_web_search.py`, `tests/test_grounding_wiring.py`, `tests/test_engine_followups.py` | tests |

---

## Task 1: Web search tools

**Files:** Create `cosci/tools/__init__.py`, `cosci/tools/web_search.py`; Test `tests/test_web_search.py`

**Interfaces / behavior:**
- `Article` (dataclass): `title:str`, `summary:str`, `url:str=""`, `published:str=""`.
- `WebSearchBackend` (Protocol): `async search(self, query:str, max_results:int=5) -> list[Article]`.
- `format_articles(articles) -> str`: `""` if empty; else one block per article: `f"- {a.title} ({a.published}) [{a.url}]\n  {a.summary}"`, joined by `"\n"`.
- `ArxivBackend(client=None)`: `async search(query, max_results=5)`: if `self._client is None`, lazy `import arxiv; self._client = arxiv.Client()`. Build `arxiv.Search(query=query, max_results=max_results)` (import inside method); run `list(self._client.results(search))` in `await asyncio.to_thread(...)`; map each result `r` → `Article(title=r.title, summary=r.summary, url=r.entry_id, published=str(r.published))`. (The injectable `client` must expose `.results(search) -> iterable` with `.title/.summary/.entry_id/.published` — tests inject a fake.)
- `TavilyBackend(api_key)`: `async search(...)`: lazy `import httpx`; POST `https://api.tavily.com/search` with the key + query; map results to `Article`. (Needs a key; not unit-tested live — only construction is.)
- `build_backend(cfg) -> WebSearchBackend | None`: `b = cfg.grounding.backend`; `"arxiv"`→`ArxivBackend()`; `"none"`→`None`; `"tavily"`→`TavilyBackend(os.environ["WEB_SEARCH_API_KEY"])` (raise a clear `RuntimeError` if the key is absent); default→`ArxivBackend()`.
- `is_faithful_grounding(cfg) -> bool`: `cfg.grounding.backend in {"tavily","serper","brave"}`.

- [ ] **Step 1: Failing test** — `tests/test_web_search.py`:
```python
import pytest
from cosci.tools.web_search import (Article, ArxivBackend, format_articles,
                                     build_backend, is_faithful_grounding)
from cosci.config import load_config

def test_format_articles_empty_and_nonempty():
    assert format_articles([]) == ""
    out = format_articles([Article(title="T", summary="S", url="u", published="2024")])
    assert "T" in out and "S" in out and "u" in out and "2024" in out

class _FakeResult:
    def __init__(self, t): self.title=t; self.summary="abs "+t; self.entry_id="http://x/"+t; self.published="2024-01-01"
class _FakeClient:
    def results(self, search): return [_FakeResult("a"), _FakeResult("b")]

@pytest.mark.asyncio
async def test_arxiv_backend_parses_injected_client():
    arts = await ArxivBackend(client=_FakeClient()).search("quantum", max_results=2)
    assert [a.title for a in arts] == ["a", "b"]
    assert arts[0].summary == "abs a" and arts[0].url == "http://x/a"

def test_build_backend_and_faithful_flag():
    cfg = load_config("config.yaml")  # default grounding.backend == "arxiv"
    assert isinstance(build_backend(cfg), ArxivBackend)
    assert is_faithful_grounding(cfg) is False
    cfg.grounding.backend = "none"
    assert build_backend(cfg) is None
    cfg.grounding.backend = "tavily"
    assert is_faithful_grounding(cfg) is True
```

- [ ] Steps 2–4 (FAIL → implement → PASS, from repo root). **Step 5: Commit** `feat(tools): pluggable web-search backends (arxiv default) + grounding helpers`.

---

## Task 2: Wire grounding into generation + reflection

**Files:** Modify `cosci/agents/generation.py`, `cosci/agents/reflection.py`; Test `tests/test_grounding_wiring.py`

**Behavior:**
- `GenerationAgent(strategies=None, grounding=None)`: in the `literature_review` strategy, if `self.grounding is not None`: `articles = await self.grounding.search(goal, max_results=5)`; `articles_block = format_articles(articles)`; else `""`. Pass that as `articles_with_reasoning`. (Other strategies unchanged — they already pass `""`.)
- `ReflectionAgent(grounding=None)`: in the FULL review, if `self.grounding is not None`: `articles = await self.grounding.search(hypothesis.text[:300], max_results=5)`; `articles_block = format_articles(articles)`; else `""`. Pass as `articles_with_reasoning`.
- Both keep working with `grounding=None` (Phase 3 behavior). Import `format_articles` from `cosci.tools.web_search`.

- [ ] **Step 1: Failing test** — `tests/test_grounding_wiring.py`:
```python
import pytest
from cosci.agents.generation import GenerationAgent
from cosci.agents.reflection import ReflectionAgent
from cosci.tools.web_search import Article
from cosci.memory import ContextMemory
from cosci.models import Hypothesis, ResearchPlan, Task, AgentName, TaskType
from cosci.config import load_config
from tests.fake_llm import FakeLLM

class _Backend:
    def __init__(self): self.queries = []
    async def search(self, query, max_results=5):
        self.queries.append(query)
        return [Article(title="Paper1", summary="relevant finding", url="u1", published="2024")]

@pytest.mark.asyncio
async def test_generation_passes_grounded_articles_into_prompt():
    cfg = load_config("config.yaml")
    mem = ContextMemory(research_plan=ResearchPlan(goal="cure X", preferences=["novel"]))
    backend = _Backend()
    captured = {}
    def router(a, m):
        captured["prompt"] = m[-1]["content"]
        return "Hypothesis: grounded idea."
    agent = GenerationAgent(strategies=["literature_review"], grounding=backend)
    await agent.execute(Task(agent=AgentName.GENERATION, action=TaskType.CREATE_INITIAL_HYPOTHESES),
                        mem, FakeLLM(router), cfg)
    assert backend.queries == ["cure X"]                 # searched on the goal
    assert "Paper1" in captured["prompt"]                # article flowed into the rendered prompt

@pytest.mark.asyncio
async def test_reflection_grounds_full_review_on_hypothesis():
    cfg = load_config("config.yaml")
    mem = ContextMemory(research_plan=ResearchPlan(goal="g"))
    mem.add_hypothesis(Hypothesis(id="G1", text="some hypothesis text", title="T", source_strategy="s"))
    backend = _Backend()
    captured = {}
    def router(a, m):
        c = m[-1]["content"]; captured.setdefault("prompts", []).append(c)
        return "verification: verified" if "deep verification" in c else "safety: safe"
    await ReflectionAgent(grounding=backend).execute(
        Task(agent=AgentName.REFLECTION, action=TaskType.REVIEW_HYPOTHESIS, target_id="G1"),
        mem, FakeLLM(router), cfg)
    assert backend.queries and backend.queries[0].startswith("some hypothesis")
    assert any("Paper1" in p for p in captured["prompts"])

@pytest.mark.asyncio
async def test_generation_no_backend_still_works():
    cfg = load_config("config.yaml")
    mem = ContextMemory(research_plan=ResearchPlan(goal="g", preferences=[]))
    res = await GenerationAgent(strategies=["literature_review"]).execute(
        Task(agent=AgentName.GENERATION, action=TaskType.CREATE_INITIAL_HYPOTHESES),
        mem, FakeLLM(lambda a, m: "Hypothesis: x"), cfg)
    assert len(res.new_hypotheses) == 1     # parametric fallback unchanged
```

- [ ] Steps 2–4 (FAIL → implement → PASS + full suite from repo root). **Step 5: Commit** `feat(agents): wire literature grounding into generation + reflection`.

---

## Task 3: Phase 4 follow-up fixes

**Files:** Modify `cosci/engine.py`; Test `tests/test_engine_followups.py`

**Behavior:**
1. **round_based evolved-review chaining:** in `round_based` mode, when an agent step returns follow-up REVIEW_HYPOTHESIS tasks (notably from EVOLVE_TOP and from generation), process them through `_run_one` (review → which then emits ADD_TO_TOURNAMENT, also processed) so evolved/new hypotheses are reviewed + safety-vetted, not left UNREVIEWED. Reuse the chaining helper; bound by the round loop. (Keep it simple: after each `_run_one`, drain the returned follow-ups by recursively running REVIEW_HYPOTHESIS and ADD_TO_TOURNAMENT tasks, applying `manage_follow_ups` for the quarantine.)
2. **Overview empty-guard:** in `run_engine`, distinguish unsafe-abort from a completed-but-empty overview. Unsafe-abort returns `None` (unchanged). On a completed run, if the meta-review overview is falsy, return `""` (empty string), never `None`. So callers can test `overview is None` for abort vs `overview == ""`/text for completion.

- [ ] **Step 1: Failing test** — `tests/test_engine_followups.py`:
```python
import pytest
from cosci.engine import run_engine
from cosci.memory import ContextMemory
from cosci.config import load_config
from cosci.models import AgentName, Safety
from cosci.agents import (GenerationAgent, ReflectionAgent, RankingAgent,
                          EvolutionAgent, ProximityAgent, MetaReviewAgent)
from tests.fake_llm import FakeLLM

class _Enc:
    def encode(self, texts): return [[float(len(t)), 1.0] for t in texts]

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
    if "preferences, attributes" in c or "structured research plan" in c:
        return '{"goal": "g", "preferences": [], "attributes": [], "constraints": []}'
    if "deep verification" in c: return "verification: verified"
    if "better idea" in c or "comparative" in c: return "better idea: 1"
    if "meta-analysis" in c or "research overview" in c or "research directions" in c: return "Overview text."
    if "safety:" in c or "review" in c: return "safety: safe"
    return "Hypothesis: mechanism."

@pytest.mark.asyncio
async def test_round_based_reviews_evolved_hypotheses():
    cfg = load_config("config.yaml"); cfg.budget.max_ideas = 3; cfg.budget.max_matches_per_idea = 1
    mem = ContextMemory()
    ov = await run_engine("g", mem, FakeLLM(_router), cfg, _agents(), mode="round_based")
    assert isinstance(ov, str)
    evolved = [h for h in mem.hypotheses.values() if h.origin.value == "evolved"]
    # every evolved hypothesis got reviewed (safety set away from the unreviewed default)
    assert all(h.safety != Safety.UNREVIEWED for h in evolved)

@pytest.mark.asyncio
async def test_overview_empty_guard_not_none_on_completion():
    cfg = load_config("config.yaml"); cfg.budget.max_ideas = 1; cfg.budget.max_matches_per_idea = 1
    mem = ContextMemory()
    def router(a, m):
        c = m[-1]["content"].lower()
        if "safety" in c and "research goal" in c: return "safety: safe"
        if "preferences, attributes" in c: return '{"goal":"g","preferences":[],"attributes":[],"constraints":[]}'
        if "research overview" in c or "research directions" in c or "meta-analysis" in c: return ""  # empty overview
        if "deep verification" in c: return "verification: verified"
        if "safety:" in c or "review" in c: return "safety: safe"
        return "Hypothesis: x"
    ov = await run_engine("g", mem, FakeLLM(router), cfg, _agents(), mode="continuous")
    assert ov == ""           # completed run with empty overview → "" not None
```

- [ ] Steps 2–4 (FAIL → implement → PASS + full suite from repo root). **Step 5: Commit** `fix(orchestration): review evolved hypotheses in round_based; overview empty-guard`.

---

## Phase 5 done-criteria
- Full suite green from repo root; grounding flows into generation+reflection when a backend is injected and falls back cleanly when not; arXiv backend parses an injected client with no network; round_based reviews evolved hypotheses; overview returns `""` (not `None`) on an empty completed run.

## Self-review notes
- **Spec coverage:** spec §11 (tools/grounding, arxiv default + pluggable, faithful-grounding flag) → Tasks 1–2; the two Phase 4 final-review follow-ups → Task 3.
- **Deferred (documented):** real Tavily/Serper/Brave live calls are unit-tested only for construction/selection (no key in CI); the CLI startup grounding warning + `run_config` faithful flag land in Phase 6.
- **No network in tests:** `ArxivBackend` injectable client; `arxiv`/`httpx` lazy-imported.
- **Type consistency:** `WebSearchBackend.search`, `Article`, `format_articles`, `build_backend`, `is_faithful_grounding`, and the `grounding=None` agent params reused as named; agents/engine from Phases 3–4.
