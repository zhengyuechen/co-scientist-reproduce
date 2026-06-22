# Co-Scientist Reproduction — Design Spec

- **Date:** 2026-06-22
- **Status:** Approved design, pending implementation plan
- **Source paper:** Gottweis, Weng, Daryin, Tu, et al., "Accelerating scientific discovery with Co-Scientist," *Nature* (Accelerated Article Preview, s41586-026-10644-y, accepted May 2026). PDFs in `research-agents/`.
- **Goal in one line:** A runnable, local, faithful reproduction of the Co-Scientist multi-agent architecture and behavior — async Supervisor/worker-pool, six specialized agents, Elo scientific-debate tournament, evolution, meta-review feedback loop, persistent context memory, and test-time compute scaling.

---

## 1. Goal & scope

### In scope (faithful reproduction)
- The **six specialized agents**: Generation, Reflection, Ranking, Evolution, Proximity, Meta-review, plus the **Supervisor**.
- The **asynchronous task-execution framework**: a `GlobalTaskQueue`, a worker pool draining it concurrently, agents enqueuing follow-up tasks via the Supervisor, and `DecideNextSteps` when the queue idles (the SN8 pseudocode mechanism).
- The **Elo tournament** via LLM **scientific debate** (multi-turn for top hypotheses, single-turn for the rest), initial rating 1200, Proximity-graph-guided match selection.
- The **evolution** strategies and the **meta-review feedback loop** (system feedback appended to every agent's prompt next iteration — "learning without back-propagation").
- **Persistent context memory** with JSON snapshots → restartable runs.
- **Test-time compute scaling**: more budget → more generation/tournament/evolution → rising Elo, observable from logs.
- **Grounding**: arXiv by default (zero-config), pluggable web-search backend. ⚠️ **Fidelity note:** the paper grounds on broad web/literature search, *not* arXiv alone. arXiv-only is a local convenience and a **fidelity compromise**; a faithful run should configure a broader web-search backend (see §11).
- **Backend**: OpenRouter, per-agent configurable model.
- **Interface**: CLI + structured console/JSON/markdown run logs.
- **Minimal expert-in-the-loop**: at startup the scientist may seed user-supplied hypotheses and extra constraints (`--seed-hypotheses`, `--constraints`); a run may be resumed from a snapshot with an appended feedback note injected like meta-review `SystemWideFeedback`. This is the *minimal* faithful nod to the paper's scientist-in-the-loop paradigm (see the non-goal on live steering).

### Non-goals (cannot or will not reproduce)
- Google-internal infrastructure; the exact Gemini base model (we are model-agnostic via OpenRouter — the paper itself states the architecture is model-agnostic).
- Specialized validation tooling (AlphaFold/ESMFold/RoseTTAFold, Open Targets/TCGA/DepMap pipelines). These are optional future stubs, not part of this build.
- The biomedical wet-lab validations from the paper.
- **Live, interactive mid-run steering** (real-time natural-language chat that redirects the system while it runs). The paper emphasizes this; v1 supports only the *startup seeding* + *resume-with-feedback* path (in-scope above), not live steering. Marked out of scope to bound the build.
- 1:1 numeric reproduction of results (impossible: different base model, different scale, and several hyperparameters/prompts are not published).

### Fidelity statement
**Faithful where the SM specifies; transparently reconstructed where it does not.** Every prompt and constant is tagged with its provenance: `SM-verbatim`, `SM-pseudocode`, or `RECONSTRUCTED`/`OURS`. The reproduction aims to match the paper's *described behavior*, not to claim bit-identical prompts the authors never released. The paper's source code is **not** released; SN8 (pseudocode) + SN9 (prompts) are the authors' stated reproducibility aids and are our ground truth.

---

## 2. Decisions locked in brainstorming

| Axis | Decision |
|---|---|
| LLM backend | **OpenRouter**, per-agent configurable model |
| Grounding | **arXiv default**, pluggable web-search backend (Tavily/Serper/Brave by key) |
| Interface | **CLI** + structured run logs (JSON + markdown) |
| Scheduler | **Continuous async** (default) with `scheduler: round_based` config flag |
| Context memory | **JSON snapshot**, restartable |
| Agent/prompt fidelity | **Full agent set**; verbatim SM prompts where given, reconstructed-in-style (labeled) where not |

---

## 3. Architecture

```
research-agents/co-scientist-reproduce/
  cosci/
    config.py        # load config.yaml + .env → typed config dataclasses
    llm.py           # OpenRouter client; per-agent model routing; retry/backoff; JSON-repair helper
    models.py        # Hypothesis, Review, ResearchPlan, Task, MatchResult, RunStats (pydantic)
    memory.py        # ContextMemory (== SM "SharedMemory"); JSON snapshot save/load
    tasks.py         # Task, TaskType, GlobalTaskQueue (asyncio priority queue)
    supervisor.py    # StartCoScientist / ManageFollowUpTasks / DecideNextSteps / summary stats /
                     #   agent-sampling weights / termination check / safety gate
    scheduler.py     # async worker pool (continuous) + round_based mode (flag)
    elo.py           # Elo update (OURS: K=32, 400-scale) + rating helpers
    agents/
      base.py        # Agent protocol: execute(task, memory, llm) -> Results
      generation.py  # strategies: literature_review, scientific_debate(selfplay),
                     #   iterative_assumptions*, research_expansion*
      reflection.py  # reviews: initial, full(+search), deep_verification, observation, simulation*, recurrent*
      ranking.py     # tournament: AddToTournament, RunTournamentBatch; single-turn + multi-turn debate match
      evolution.py   # strategies: combine, simplify, feasibility, out_of_box, grounding*, inspiration*
      proximity.py   # embedding similarity graph (sentence-transformers MiniLM)
      meta_review.py # meta-analysis synthesis, research_overview, system_feedback
    prompts/
      verbatim.py        # EXACT SN9 prompts (each cited to SN9.x)
      reconstructed.py   # in-style prompts for SM gaps (each tagged RECONSTRUCTED + Methods intent)
    tools/
      web_search.py  # WebSearchBackend protocol; arxiv default; optional Tavily/Serper/Brave
      arxiv_tool.py  # arXiv query/fetch (free, no key)
    logging_utils.py # structured console + JSON/markdown run-log writers
    cli.py           # entrypoint: research goal → run engine → stream → results
  config.yaml        # models per agent, budgets, thresholds, scheduler mode, grounding backend
  .env.example       # OPENROUTER_API_KEY, optional WEB_SEARCH_API_KEY
  .gitignore         # .env, results/, snapshots/, __pycache__
  requirements.txt
  README.md
  docs/2026-06-22-co-scientist-reproduce-design.md   # this spec
  results/           # timestamped run outputs (gitignored)
  snapshots/         # JSON context-memory snapshots (gitignored)
  tests/
    fake_llm.py      # deterministic canned-response LLM for tests
    test_*.py
```

`*` = strategy/review whose prompt is **reconstructed** (not verbatim in SM).

**Module responsibilities (each unit has one purpose, testable in isolation):**
- `llm.py` — the only place that talks to OpenRouter. Given an agent name + messages, returns text. Handles retries, backoff, and JSON repair. Swappable with `FakeLLM` in tests.
- `memory.py` — the single source of truth for run state; pure data + (de)serialization. No LLM calls.
- `tasks.py` — queue mechanics only.
- `supervisor.py` — all orchestration *policy* (what task comes next), no agent internals.
- `scheduler.py` — all orchestration *mechanism* (concurrency), no policy.
- `agents/*` — each agent: take a task + memory + llm, do its job, return Results + follow-up task requests. No direct queue access (returns requests; Supervisor enqueues). **Implementation abstraction (labeled):** SN8 sometimes shows agents enqueuing follow-up tasks *directly* (Generation→Reflection, Reflection→Ranking). We funnel all enqueues through `Supervisor.ManageFollowUpTasks` instead — behaviorally equivalent to SN8, chosen for single-writer queue discipline and testability. This is our abstraction, not a behavior change.
- `elo.py`, `proximity.py`, `tools/*` — pure utilities.

---

## 4. Core engine & data flow (faithful to SN8)

1. CLI → `Supervisor.StartCoScientist(goal)`:
   - Run **safety review** on the goal (gate; from SM safety notes). Abort with message if unsafe.
   - Parse goal → **ResearchPlan** (`preferences`, `attributes`, `constraints`).
   - Save ResearchPlan to ContextMemory.
   - Enqueue `Task(Generation, "CreateInitialHypotheses")`.
2. **Scheduler** spins up N async workers (default 4) that loop: fetch task → run agent → hand Results to `Supervisor.ManageFollowUpTasks` → enqueue follow-ups.
3. **Task chaining** (`ManageFollowUpTasks`, from SN8):
   - `NewHypothesisCreated` → enqueue `Reflection:ReviewHypothesis`.
   - `ReviewCompleted` → **if the hypothesis passed safety review**, enqueue `Ranking:AddToTournament`. **If flagged unsafe, quarantine it** (`active=false`, `safety="unsafe"`, record reason) and do **not** enqueue it — unsafe hypotheses never enter the tournament (matches the supplement's safety mechanism, which excludes unsafe hypotheses from the tournament). Quarantined hypotheses are written to results for transparency but excluded from ranking/evolution/overview.
4. **Idle policy** (`DecideNextSteps`, from SN8, fired when queue empties / on a tick):
   - Always: enqueue `Ranking:RunTournamentBatch`.
   - If hypothesis quality plateaued → `Evolution:EvolveTopHypotheses` (top 5).
   - Periodically → `MetaReview:GenerateSystemFeedback` (feedback appended to agent prompts next round).
   - Periodically → `MetaReview:GenerateFinalResearchOverview` (top 10) → emitted to results.
5. **Proximity** graph rebuilt periodically (embeddings) → informs tournament match selection.
6. **Termination:** the SN8 guard is `WHILE NumIdeas < MaxIdeas AND MatchesPerIdea < MaxMatches`, so the run **terminates when *either* cap is reached** (`NumIdeas ≥ MaxIdeas` **OR** `MatchesPerIdea ≥ MaxMatches`), or when the wall-clock/token budget is exhausted. `MatchesPerIdea` is ambiguous in SN8; we define it as **`min` matches_played across active hypotheses** (i.e., stop once every idea has been compared enough) — labeled OURS in §16.
7. Periodic **JSON snapshot** of ContextMemory every K completed tasks → restart on failure.
8. **Output:** timestamped run dir with hypotheses, reviews, full tournament/Elo trajectory, and the final research overview (markdown + JSON).

The SN8 reference pseudocode (Supervisor `StartCoScientist` / `ManageFollowUpTasks` / `DecideNextSteps`, and each agent function) is reproduced in **Appendix B** and is the implementation's contract.

---

## 5. Agents (detailed)

For every agent: it receives a `Task`, reads/writes `ContextMemory`, calls the LLM via `llm.py`, and **returns** Results + follow-up requests (it does not enqueue directly).

### 5.1 Generation
- **CreateInitialHypotheses** runs multiple strategies:
  - `literature_review` — arXiv/web search → synthesize → propose hypothesis. **Prompt: SM-verbatim (SN9.1 #1).**
  - `scientific_debate` (self-play, 3–5 turns, max 10) → refined hypothesis after "HYPOTHESIS" sentinel. **Prompt: SM-verbatim (SN9.1 #2).**
  - `iterative_assumptions` — conditional-reasoning hops → aggregate to hypothesis. **Prompt: RECONSTRUCTED** (Methods p.25).
  - `research_expansion` — use Meta-review overview to find unexplored areas. **Prompt: RECONSTRUCTED** (Methods p.25).
- Each new hypothesis → `NewHypothesisCreated` Results → Reflection review.

### 5.2 Reflection
Six review types (Methods pp.26–27):
- `initial` — correctness/quality/novelty/safety, **no tools** (fast discard). **RECONSTRUCTED.**
- `full` — same dims + web-search grounding. **RECONSTRUCTED** (SN10 shows outputs, not prompt).
- `deep_verification` — decompose into assumptions → sub-assumptions → evaluate each. **RECONSTRUCTED** (pseudocode one-liner in SN8).
- `observation` — does the hypothesis novelly explain article observations? **Prompt: SM-verbatim (SN9.2).**
- `simulation` — step-wise mechanism simulation to find failure modes. **RECONSTRUCTED.**
- `recurrent` — re-review using tournament results + meta-review feedback. **RECONSTRUCTED.**
- Every review also emits a **safety verdict** (`safe` / `unsafe` + reason); a hypothesis flagged unsafe by any review is marked unsafe in `ReviewCompleted`, so the Supervisor quarantines it before `AddToTournament` (§4 step 3, §12).
- Default pipeline per hypothesis: `full` + `deep_verification` (matching SN8 `ReviewHypothesis`); other types configurable. → `ReviewCompleted` Results.

### 5.3 Ranking (tournament)
- `AddToTournament` — set Elo=1200 if not already in tournament (idempotent). **SM-pseudocode.**
- `RunTournamentBatch` — select a pair, run a debate match, update Elo.
  - **Match selection** (Methods p.26): prefer comparing *similar* hypotheses (Proximity graph) and prioritize *new* + *top-ranked* hypotheses.
  - **Top-ranked pair** → `multi_turn` scientific debate (3–5 turns, max 10). **Prompt: SM-verbatim (SN9.3 #2).**
  - **Lower-ranked pair** → `single_turn` comparison. **Prompt: SM-verbatim (SN9.3 #1).**
  - Each match ends with `better idea: <1 or 2>` → winner/loser → `elo.update`.

### 5.4 Evolution
- `EvolveTopHypotheses` on top 5; strategies:
  - `feasibility` improvement. **Prompt: SM-verbatim (SN9.4 #1).**
  - `out_of_box` (analogy/inspiration, single new hypothesis). **Prompt: SM-verbatim (SN9.4 #2).**
  - `combine` two top hypotheses. **RECONSTRUCTED** (SN8 one-liner).
  - `simplify` for testability. **RECONSTRUCTED** (SN8 one-liner).
  - `grounding` enhancement, `inspiration` from existing. **RECONSTRUCTED** (Methods p.27).
- Evolved hypotheses are treated as new → reviewed → enter tournament (SN8).

### 5.5 Proximity
- `UpdateProximityGraph` — pairwise similarity over all hypotheses via **sentence-transformers** embeddings (MiniLM); store adjacency. Optional LLM 1–5 similarity scoring as a config alternative (SM mentions a 1–5 scale in ablation). **Method only in SM.**

### 5.6 Meta-review
- `GenerateSystemFeedback` — synthesize all reviews + debate transcripts into common strengths/weaknesses → `SystemWideFeedback` (appended to agent prompts next round). **Synthesis prompt: SM-verbatim (SN9.5);** the SN8 feedback variant is **RECONSTRUCTED**.
- `GenerateFinalResearchOverview` — synthesize top 10 into a coherent research overview. **RECONSTRUCTED** (SN8 one-liner).

---

## 6. Prompts

- `prompts/verbatim.py` holds the **exact SN9 prompts** (Appendix A), each as a constant with an `# SN9.x` citation and the verbatim `{placeholders}` (including SN9.3's inconsistent `{hypothesis 1}` / `{review1}` spellings). A small documented `PLACEHOLDER_MAP` defines the normalization from those verbatim names to the Python identifiers used at render time, so the stored prompt text stays byte-faithful to the PDF.
- `prompts/reconstructed.py` holds in-style prompts for every gap, each with:
  ```python
  # RECONSTRUCTED — not from SM. Methods intent: <one line>. Style-matched to SN9.x.
  ```
- Placeholder population (`{goal}`, `{preferences}`, `{instructions}`, `{idea_attributes}`, `{notes}`, `{reviews_overview}`, `{articles_with_reasoning}`, `{transcript}`, `{hypothesis}`, `{review}`) is centralized in a small `prompts/render.py` helper so the assembly logic (including how meta-review feedback maps into `{instructions}`/`{notes}`) is one well-tested place. (Assembly logic is an SM gap — OURS.)

---

## 7. Tournament & Elo

- Initial rating **1200** (SM). Update: **OURS** — standard Elo `R' = R + K·(S − E)`, `E = 1/(1+10^((R_opp−R)/400))`, **K=32**, winner S=1 / loser S=0. Clearly labeled as our default (SM gives no K/formula).
- Match selection uses the Proximity graph (compare similar) and prioritizes new + top-ranked hypotheses.
- `matches_played` tracked per hypothesis; `MatchesPerIdea` for termination = `min` over active hypotheses (§4).
- **Verdict parsing:** the match decision is parsed from the model's final line; the parser accepts **both** `better idea: <1|2>` and `better hypothesis: <1|2>` (the verbatim SN9.3 single-turn prompt uses both phrasings inconsistently — preserved as-is in Appendix A.4). Parse failure → re-ask once, then count the match as a draw with a logged warning (no Elo change).

---

## 8. Context memory schema (JSON-serializable)

```
ContextMemory:
  research_plan: {goal, preferences[], attributes[], constraints[]}
  hypotheses: { id -> Hypothesis{ id, text, title, source_strategy, parent_ids[],
                                  elo_rating|null, matches_played, active, created_tick,
                                  safety: "unreviewed"|"safe"|"unsafe", safety_reason|null,
                                  origin: "generated"|"evolved"|"user_seed" } }
  reviews:    { hypothesis_id -> [ Review{ type, scores{}, text, references[], tool_grounded } ] }
  tournament: [ MatchResult{ a_id, b_id, mode, winner_id, elo_before{}, elo_after{}, transcript, tick } ]
  proximity:  { id -> [ {other_id, similarity} ] }
  system_feedback: str                 # appended to agent prompts next round
  stats: { num_ideas, num_reviews_pending, matches_total, strategy_effectiveness{}, tick, tokens_spent }
```
- `memory.save_snapshot(path)` / `memory.load_snapshot(path)` — full JSON round-trip. Snapshot every K tasks and on clean exit. Restart = load latest snapshot, rebuild the queue from in-flight/derivable state.
- Concurrency: ContextMemory access guarded by an `asyncio.Lock` (single event loop, so coarse locking around mutations is sufficient and simple). **Concurrency model is OURS** (SM unspecified).

---

## 9. Config & hyperparameter defaults

`config.yaml` (all overridable). Provenance tagged.

| Param | Default | Provenance |
|---|---|---|
| `models.<agent>` | one cheap/free OpenRouter model; per-agent overridable | OURS (model-agnostic per paper) |
| `elo.k_factor` / `elo.scale` | 32 / 400 | OURS (SM: init=1200 only) |
| `budget.max_ideas` | 20 | OURS (SM names `MaxIdeas`, no value) |
| `budget.max_matches_per_idea` | 8 | OURS (SM names it, no value) |
| `budget.max_wallclock_s` / `budget.max_tokens` | 1800 / null | OURS (added safety cap) |
| `temperature.{gen,reflect,rank,evolve}` | 0.7 / 0.5 / 0.3 / 0.8 | OURS (SM: none) |
| `debate.turns_typical` / `debate.turns_max` | 3–5 / 10 | **SM (SN9)** |
| `evolution.top_k` | 5 | **SM (SN8)** |
| `overview.top_n` | 10 | **SM (SN8)** |
| `workers` | 4 | SM Fig 1b (illustrative) |
| `scheduler` | `continuous` (or `round_based`) | OURS (our flag) |
| `proximity.method` | `embeddings` (MiniLM) | SM ("e.g., text embeddings") |
| `grounding.backend` | `arxiv` (or `tavily`/`serper`/`brave`) | OURS |
| `plateau.window` / `plateau.epsilon` | 10 matches / small Δ best-Elo | OURS (SM: "stopped improving", no criterion) |

---

## 10. Test-time compute scaling

- Driven by the budget (`max_ideas`, `max_matches_per_idea`, wall-clock/token caps). Larger budget → the Supervisor keeps dispatching generation/tournament/evolution → Elo of top hypotheses rises (the paper's Fig 2a effect).
- Supervisor computes **summary statistics** each tick (num ideas, pending reviews, tournament progress, per-strategy effectiveness) and **weight-samples** which agent to favor in `DecideNextSteps`. **Weighting function is OURS** (SM qualitative only): start with the SN8 heuristic triggers, plus a simple weighting that up-samples Evolution when generation novelty plateaus and Generation when idea count is low.
- The CLI logs the **Elo trajectory** (top-10 mean + best) over ticks so scaling is observable, mirroring the paper's temporal-bucket analysis.

---

## 11. Tools & grounding

- `WebSearchBackend` protocol: `search(query) -> [Article{title, abstract, url, date}]` and `fetch(url) -> text`.
- Default `ArxivBackend` (free, no key) — reuse the pattern from the existing `open-ai-co-scientist` arxiv tool.
- Optional `TavilyBackend`/`SerperBackend`/`BraveBackend` enabled when `WEB_SEARCH_API_KEY` is set + `grounding.backend` selected.
- Generation `literature_review` and Reflection `full`/novelty use grounding; if no backend/result, degrade gracefully to parametric reasoning with a logged note.
- **Fidelity warning at startup:** if `grounding.backend == arxiv` (or no web backend is configured), the CLI prints a prominent warning that the run is *not* using the broad web search the paper relied on, so novelty/grounding fidelity is reduced. A run is labeled "faithful-grounding" in `run_config.json` only when a general web-search backend is active.

---

## 12. Error handling & safety

- LLM calls: `tenacity` retry with exponential backoff on transient/network/5xx; on persistent failure return a structured error that the agent turns into a logged skip (never crash the run).
- JSON outputs: strip code fences, parse; on failure re-ask **once** with a "return valid JSON only" nudge; on second failure, skip that item with a logged error.
- Per-task isolation: a failing task fails only its worker iteration; the run continues. Snapshots allow resume.
- **Safety gates (two levels, from SM safety notes; prompts RECONSTRUCTED):**
  1. **Goal gate** — Supervisor safety-reviews the research goal before any generation; unsafe → abort with explanation.
  2. **Per-hypothesis gate** — each generated/evolved hypothesis is safety-reviewed during Reflection; any hypothesis flagged unsafe is **quarantined** (`active=false`) and **excluded from the tournament, evolution, and the final overview** (the supplement excludes unsafe hypotheses from the tournament). Quarantined items are still logged for transparency.

---

## 13. Testing strategy (TDD)

- `tests/fake_llm.py` — deterministic `FakeLLM` keyed by agent/strategy returning canned, well-formed responses (incl. `better idea: 1`, `HYPOTHESIS …`, JSON reviews). Enables zero-API unit tests.
- Unit tests:
  - queue mechanics + `ManageFollowUpTasks` chaining (new→review→tournament).
  - `DecideNextSteps` triggers (plateau → evolve; tick → feedback/overview).
  - `elo.update` math (symmetry, expected-score, K).
  - tournament match-selection (prefers similar / new / top-ranked).
  - `memory` snapshot save/load round-trip (byte-stable for same state).
  - termination logic (max_ideas ∧ max_matches; budget caps).
  - prompt rendering (placeholders all filled; feedback injection).
- One optional **live smoke test** (gated by env): tiny budget (2 hypotheses, 2 matches) on a free model; asserts a run completes and produces a research overview.

---

## 14. Results & logging

```
results/<YYYY-MM-DD>_<HHMMSS>_<goal-slug>/
  run_config.json        # resolved config + provenance
  hypotheses.json        # all hypotheses with final Elo, lineage, source strategy
  reviews.json
  tournament.jsonl       # one line per match (Elo before/after, mode, transcript ref)
  elo_trajectory.csv     # tick, top10_mean_elo, best_elo  (the scaling curve)
  research_overview.md   # final synthesized overview (human-facing deliverable)
  research_overview.json
  run.log                # structured console log
```
Console streams: generation events, review verdicts, each match's Elo change, evolution events, meta-review feedback, and periodic Elo-trajectory lines.

---

## 15. Implementation phases (high-level; detailed plan comes from writing-plans)

1. **Foundation** — repo scaffold, `config`, `llm` (+ FakeLLM), `models`, `memory` (+ snapshot), `tasks`, `elo`; unit tests for memory/elo/tasks.
2. **Prompts** — `verbatim.py` (Appendix A), `reconstructed.py` (Appendix C gaps), `render.py`; tests for rendering.
3. **Agents** — generation, reflection, ranking, evolution, proximity, meta-review against FakeLLM; per-agent tests.
4. **Orchestration** — `supervisor` (chaining, DecideNextSteps, stats, weighting, termination, safety) + `scheduler` (continuous + round_based); engine integration tests with FakeLLM.
5. **Grounding** — `tools/` arxiv default + pluggable backend; wire into generation/reflection.
6. **CLI + logging + results** — entrypoint, structured logs, run-dir writers, snapshot/resume; live smoke test; README.

---

## 16. Gaps (SM-unspecified) and how we fill them

| Gap | Resolution (OURS, labeled in code) |
|---|---|
| Elo K-factor / formula | Standard Elo, K=32, 400-scale |
| `MaxIdeas`, `MaxMatchesPerIdea` | Config defaults 20 / 8 + wall-clock/token caps |
| Decoding params (temperature etc.) | Config defaults (§9) |
| Supervisor agent-sampling weights | SN8 heuristic triggers + simple plateau-aware weighting |
| Worker count / concurrency / locking | 4 workers, single event loop, coarse `asyncio.Lock` on memory |
| Missing prompts (§5 `*`) | Reconstructed in SN9 style, tagged RECONSTRUCTED |
| Proximity specifics | sentence-transformers MiniLM embeddings; optional LLM 1–5 scoring |
| Context-memory schema details | §8 JSON schema |
| Placeholder assembly logic | `prompts/render.py` |
| Plateau / "enough time" triggers | `plateau.window`/`epsilon`, periodic ticks |
| Tool interfaces | `WebSearchBackend` protocol (§11) |

---

## 17. Risks & open questions

- **Cost**: real debates/tournaments are many LLM calls. Mitigation: free/cheap default model, small default budgets, `round_based` mode for cheap runs, hard token cap.
- **Reconstructed-prompt quality**: reconstructed prompts won't perfectly match the authors'. Mitigation: style-match SN9, keep them in one labeled module for easy iteration, and (future) A/B against verbatim ones where possible.
- **Concurrency correctness**: shared ContextMemory under N workers. Mitigation: single event loop + coarse lock; integration tests on chaining.
- **Scope**: this is one coherent system but a large build (6 agents + engine). If the implementation plan proves too big for one pass, decompose by the §15 phases.

---

## Appendix A — Verbatim SM prompts (SN9)

> Reproduced exactly from SN9, **including the supplement's own inconsistent placeholder spellings** (e.g. SN9.3 uses `{hypothesis 1}` / `{review 1}` with spaces, and is internally inconsistent — `{review1}` vs `{review 2}` in the multi-turn prompt). These are the verbatim PDF strings. In code, `prompts/verbatim.py` keeps these exact strings; a documented `PLACEHOLDER_MAP` normalizes them to valid Python identifiers (`{hypothesis_1}`, `{review_1}`, …) only at render time, so the stored prompt text stays byte-faithful to the PDF. Source: SN9 of the supplementary information.

### A.1 Generation — after literature review (SN9.1 #1)
```
You are an expert tasked with formulating a novel and robust hypothesis to address the following objective.
Describe the proposed hypothesis in detail, including specific entities, mechanisms, and anticipated outcomes.
This description is intended for an audience of domain experts.
You have conducted a thorough review of relevant literature and developed a logical framework for addressing the objective. The articles consulted, along with your analytical reasoning, are provided below.

Goal: {goal}

Criteria for a strong hypothesis:
{preferences}

Existing hypothesis (if applicable):
{source_hypothesis}

{instructions}

Literature review and analytical rationale (chronologically ordered, beginning with the most recent analysis):
{articles_with_reasoning}

Proposed hypothesis (detailed description for domain experts):
```

### A.2 Generation — after scientific debate / self-play (SN9.1 #2)
```
You are an expert participating in a collaborative discourse concerning the generation of a {idea_attributes} hypothesis. You will engage in a simulated discussion with other experts. The overarching objective of this discourse is to collaboratively develop a novel and robust {idea_attributes} hypothesis.

Goal: {goal}

Criteria for a high-quality hypothesis:
{preferences}

Instructions:
{instructions}

Review Overview:
{reviews_overview}

Procedure:
Initial contribution (if initiating the discussion):
Propose three distinct {idea_attributes} hypotheses.

Subsequent contributions (continuing the discussion):
* Pose clarifying questions if ambiguities or uncertainties arise.
* Critically evaluate the hypotheses proposed thus far, addressing the following aspects:
    - Adherence to {idea_attributes} criteria.
    - Utility and practicality.
    - Level of detail and specificity.
* Identify any weaknesses or potential limitations.
* Propose concrete improvements and refinements to address identified weaknesses.
* Conclude your response with a refined iteration of the hypothesis.

General guidelines:
* Exhibit boldness and creativity in your contributions.
* Maintain a helpful and collaborative approach.
* Prioritize the generation of a high-quality {idea_attributes} hypothesis.

Termination condition:
When sufficient discussion has transpired (typically 3-5 conversational turns, with a maximum of 10 turns) and all relevant questions and points have been thoroughly addressed and clarified, conclude the process by writing "HYPOTHESIS" (in all capital letters) followed by a concise and self-contained exposition of the finalized idea.

#BEGIN TRANSCRIPT#
{transcript}
#END TRANSCRIPT#

Your Turn:
```

### A.3 Reflection — observation review (SN9.2)
```
You are an expert in scientific hypothesis evaluation. Your task is to analyze the relationship between a provided hypothesis and observations from a scientific article. Specifically, determine if the hypothesis provides a novel causal explanation for the observations, or if they contradict it.

Instructions:
1. Observation extraction: list relevant observations from the article.
2. Causal analysis (individual): for each observation:
   a. State if its cause is already established.
   b. Assess if the hypothesis could be a causal factor (hypothesis => observation). Start with: "would we see this observation if the hypothesis was true:".
   c. Explain if it's a novel explanation. If not, or if a better explanation exists, state: "not a missing piece."
3. Causal analysis (summary): determine if the hypothesis offers a novel explanation for a subset of observations. Include reasoning. Start with: "would we see some of the observations if the hypothesis was true:".
4. Disproof analysis: determine if any observations contradict the hypothesis. Start with: "does some observations disprove the hypothesis:".
5. Conclusion: state: "hypothesis: <already explained, other explanations more likely, missing piece, neutral, or disproved>".

Scoring:
* Already explained: hypothesis consistent, but causes are known. No novel explanation.
* Other explanations more likely: hypothesis *could* explain, but better explanations exist.
* Missing piece: hypothesis offers a novel, plausible explanation.
* Neutral: hypothesis neither explains nor is contradicted.
* Disproved: observations contradict the hypothesis.

Important: if observations are expected regardless of the hypothesis, and don't disprove it, it's neutral.

Article:
{article}

Hypothesis:
{hypothesis}

Response (provide reasoning. end with: "hypothesis: <already explained, other explanations more likely, missing piece, neutral, or disproved>".)
```

### A.4 Ranking — single-turn comparison (SN9.3 #1)
```
You are an expert evaluator tasked with comparing two hypotheses.
Evaluate the two provided hypotheses (hypothesis 1 and hypothesis 2) and determine which one is superior based on the specified {idea_attributes}.
Provide a concise rationale for your selection, concluding with the phrase "better idea: <1 or 2>".

Goal: {goal}

Evaluation criteria:
{preferences}

Considerations:
{notes}
Each hypothesis includes an independent review. These reviews may contain numerical scores. Disregard these scores in your comparative analysis, as they may not be directly comparable across reviews.

Hypothesis 1:
{hypothesis 1}

Hypothesis 2:
{hypothesis 2}

Review of hypothesis 1:
{review 1}

Review of hypothesis 2:
{review 2}

Reasoning and conclusion (end with "better hypothesis: <1 or 2>"):
```

### A.5 Ranking — multi-turn scientific debate (SN9.3 #2)
```
You are an expert in comparative analysis, simulating a panel of domain experts engaged in a structured discussion to evaluate two competing hypotheses. The objective is to rigorously determine which hypothesis is superior based on a predefined set of attributes and criteria. The experts possess no pre-existing biases toward either hypothesis and are solely focused on identifying the optimal choice, given that only one can be implemented.

Goal: {goal}

Criteria for hypothesis superiority:
{preferences}

Hypothesis 1:
{hypothesis 1}

Hypothesis 2:
{hypothesis 2}

Initial review of hypothesis 1:
{review1}

Initial review of hypothesis 2:
{review 2}

Debate procedure:
The discussion will unfold in a series of turns, typically ranging from 3 to 5, with a maximum of 10.
Turn 1: begin with a concise summary of both hypotheses and their respective initial reviews.
Subsequent turns:
* Pose clarifying questions to address any ambiguities or uncertainties.
* Critically evaluate each hypothesis in relation to the stated Goal and Criteria. This evaluation should consider aspects such as:
    - Potential for correctness/validity.
    - Utility and practical applicability.
    - Sufficiency of detail and specificity.
    - Novelty and originality.
    - Desirability for implementation.
* Identify and articulate any weaknesses, limitations, or potential flaws in either hypothesis.

Additional notes:
{notes}

Termination and judgment:
Once the discussion has reached a point of sufficient depth (typically 3-5 turns, up to 10 turns) and all relevant questions and concerns have been thoroughly addressed, provide a conclusive judgment. This judgment should succinctly state the rationale for the selection. Then, indicate the superior hypothesis by writing the phrase "better idea: ", followed by "1" (for hypothesis 1) or "2" (for hypothesis 2).
```

### A.6 Evolution — feasibility improvement (SN9.4 #1)
```
You are an expert in scientific research and technological feasibility analysis. Your task is to refine the provided conceptual idea, enhancing its practical implementability by leveraging contemporary technological capabilities. Ensure the revised concept retains its novelty, logical coherence, and specific articulation.

Goal: {goal}

Guidelines:
1. Begin with an introductory overview of the relevant scientific domain.
2. Provide a concise synopsis of recent pertinent research findings and related investigations, highlighting successful methodologies and established precedents.
3. Articulate a reasoned argument for how current technological advancements can facilitate the realization of the proposed concept.
4. CORE CONTRIBUTION: Develop a detailed, innovative, and technologically viable alternative to achieve the objective, emphasizing simplicity and practicality.

Evaluation Criteria:
{preferences}

Original Conceptualization:
{hypothesis}

Response:
```

### A.7 Evolution — out-of-the-box thinking (SN9.4 #2)
```
You are an expert researcher tasked with generating a novel, singular hypothesis inspired by analogous elements from provided concepts.

Goal: {goal}

Instructions:
1. Provide a concise introduction to the relevant scientific domain.
2. Summarize recent findings and pertinent research, highlighting successful approaches.
3. Identify promising avenues for exploration that may yield innovative hypotheses.
4. CORE HYPOTHESIS: Develop a detailed, original, and specific single hypothesis for achieving the stated goal, leveraging analogous principles from the provided ideas. This should not be a mere aggregation of existing methods or entities. Think out-of-the-box.

Criteria for a robust hypothesis:
{preferences}

Inspiration may be drawn from the following concepts (utilize analogy and inspiration, not direct replication):
{hypotheses}

Response:
```

### A.8 Meta-review — synthesis (SN9.5)
```
You are an expert in scientific research and meta-analysis.
Synthesize a comprehensive meta-review of provided reviews pertaining to the following research goal:

Goal: {goal}

Preferences:
{preferences}

Additional instructions:
{instructions}

Provided reviews for meta-analysis:
{reviews}

Instructions:
* Generate a structured meta-analysis report of the provided reviews.
* Focus on identifying recurring critique points and common issues raised by reviewers.
* The generated meta-analysis should provide actionable insights for researchers developing future proposals.
* Refrain from evaluating individual proposals or reviews; focus on producing a synthesized meta-analysis.

Response:
```

---

## Appendix B — SN8 pseudocode (the engine contract)

> Reproduced from SN8; the implementation's `supervisor.py`/`scheduler.py`/`agents/*` must match this control flow.

```
// Supervisor
FUNCTION StartCoScientist(ScientistResearchGoal)
  Parse goal into ResearchPlan; SAVE to SharedMemory
  CREATE Task(Generation, "CreateInitialHypotheses"); ADD to GlobalTaskQueue
  WHILE NumberOfIdeas < MaxIdeas AND NumberOfMatchesPerIdea < MaxMatchesPerIdea DO
    IF GlobalTaskQueue not empty THEN
      FETCH next Task; AgentToRun = agent for Task; Results = AgentToRun.Execute(Task)
      ManageFollowUpTasks(Results)
    ELSE
      DecideNextSteps()

FUNCTION ManageFollowUpTasks(Results)
  IF Results.Type == "NewHypothesisCreated" THEN ADD Task(Reflection, "ReviewHypothesis", Results.HypothesisID)
  IF Results.Type == "ReviewCompleted"      THEN ADD Task(Ranking,    "AddToTournament", Results.HypothesisID)

FUNCTION DecideNextSteps()
  ADD Task(Ranking, "RunTournamentBatch")
  IF quality stopped improving THEN ADD Task(Evolution, "EvolveTopHypotheses")
  IF enough time passed THEN ADD Task(Metareview, "GenerateSystemFeedback")
  IF enough time passed THEN RETURN MetaReviewAgent.GenerateFinalResearchOverview()

// Generation.CreateInitialHypotheses → web search + LLM (literature); simulate debate; ... → save + enqueue Reflection
// Reflection.ReviewHypothesis → full review (web search) + deep verification (decompose assumptions) → enqueue Ranking AddToTournament
// Ranking.AddToTournament → if EloRating empty set 1200 (idempotent)
// Ranking.RunTournamentBatch → select pair (similar/new/top) → debate → winner/loser → update Elo
// Evolution.EvolveTopHypotheses → top 5 → combine / simplify / out-of-box / ... → save + enqueue Reflection
// Proximity.UpdateProximityGraph → pairwise embedding similarity → update graph
// Metareview.GenerateSystemFeedback → analyze reviews+transcripts → SystemWideFeedback
// Metareview.GenerateFinalResearchOverview → top 10 → synthesize overview
```

---

## Appendix C — Reconstructed-prompt inventory (SM gaps to author, in SN9 style)

- Generation: `iterative_assumptions`, `research_expansion`
- Reflection: `initial`, `full`, `deep_verification`, `simulation`, `recurrent`
- Evolution: `combine`, `simplify`, `grounding`, `inspiration`
- Meta-review: `research_overview`, `system_feedback`
- Supervisor: `goal_parse` (research-goal → ResearchPlan), `safety_review` (goal + per-hypothesis)

Each will live in `prompts/reconstructed.py`, tagged `# RECONSTRUCTED — not from SM` with a one-line Methods-intent docstring.
