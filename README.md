# Co-Scientist Reproduction

A faithful, runnable reproduction of the multi-agent architecture described in
Gottweis, Weng, Daryin, Tu, et al., *"Accelerating scientific discovery with Co-Scientist,"*
**Nature** (2026), s41586-026-10644-y.

**Status:** Phases 1–5 implemented and merged (foundation, prompts, six agents, orchestration, grounding). The engine runs a full hypothesis-evolution cycle end-to-end (verified against a deterministic fake LLM). **57 tests passing** (conda env `cosci-reproduce`). Remaining: Phase 6 (CLI + structured run logs) — until then, drive it via `cosci.engine.run_engine(...)` in code.

## What's built (`cosci/`)
- **config / models / elo / tasks / llm / memory** — typed config, domain models, Elo math, async task queue, OpenRouter client (+ deterministic `FakeLLM`), snapshot-restartable context memory.
- **prompts/** — `verbatim.py` (8 byte-faithful SN9 prompts), `reconstructed.py` (16 SN9-style prompts for gaps the paper didn't publish, each tagged), `render.py`.
- **agents/** — Generation, Reflection (with safety verdict), Ranking (Elo debate tournament), Evolution, Proximity (embeddings), Meta-review.
- **supervisor.py / engine.py** — SN8 task-chaining, safety quarantine, termination, `decide_next_steps`; the queue-driven run loop (`continuous` + `round_based`).
- **tools/web_search.py** — pluggable literature grounding (arXiv default, optional Tavily).

## Documents
- **Design spec:** [docs/2026-06-22-co-scientist-reproduce-design.md](docs/2026-06-22-co-scientist-reproduce-design.md)
- **Implementation plans:** [Phase 1 — Foundation](docs/plans/2026-06-22-co-scientist-reproduce-phase1-foundation.md) · [Phase 2 — Prompts](docs/plans/2026-06-22-co-scientist-reproduce-phase2-prompts.md) · [Phase 3 — Agents](docs/plans/2026-06-22-co-scientist-reproduce-phase3-agents.md) · [Phase 4 — Orchestration](docs/plans/2026-06-22-co-scientist-reproduce-phase4-orchestration.md) · [Phase 5 — Grounding](docs/plans/2026-06-22-co-scientist-reproduce-phase5-grounding.md)

## Tests
`conda run -n cosci-reproduce pytest -q` — run from the repo root (the agent tests load `config.yaml` by relative path). 57 passing.

## Architecture
Supervisor draining a global task queue; six specialized agents —
Generation, Reflection, Ranking, Evolution, Proximity, Meta-review — with an Elo
scientific-debate tournament, a meta-review feedback loop, persistent (snapshot-restartable)
context memory, and test-time compute scaling (more budget → more tournament/evolution rounds → rising Elo).
Backend: OpenRouter (model-agnostic). Grounding: arXiv by default, pluggable web search.

The run loop is **sequential**, faithful to the supplement's SN8 `StartCoScientist` pseudocode;
true concurrent worker dispatch (the Methods' async framework) is a documented future enhancement
(it requires splitting each agent's LLM-compute from its memory-commit). A CLI + structured run
logs is the remaining Phase 6 work.

## Fidelity policy
Prompts and constants are tagged by provenance: **verbatim** from the paper's supplementary
materials (SN8 pseudocode / SN9 prompts) where published, and **reconstructed in-style**
(clearly labeled) where the paper does not publish them. The paper's source code was not
released; SN8 + SN9 are the authors' stated reproducibility aids and our ground truth.
See the design spec for the full policy and the verbatim-vs-reconstructed breakdown.
