# Co-Scientist Reproduction

A faithful, runnable reproduction of the multi-agent architecture described in
Gottweis, Weng, Daryin, Tu, et al., *"Accelerating scientific discovery with Co-Scientist,"*
**Nature** (2026), s41586-026-10644-y.

**Status:** design spec + Phase 1 implementation plan complete; implementation in progress.

## Documents
- **Design spec:** [docs/2026-06-22-co-scientist-reproduce-design.md](docs/2026-06-22-co-scientist-reproduce-design.md)
- **Phase 1 plan (Foundation):** [docs/plans/2026-06-22-co-scientist-reproduce-phase1-foundation.md](docs/plans/2026-06-22-co-scientist-reproduce-phase1-foundation.md)

## Architecture (target)
Asynchronous Supervisor + worker pool draining a global task queue; six specialized agents —
Generation, Reflection, Ranking, Evolution, Proximity, Meta-review — with an Elo
scientific-debate tournament, a meta-review feedback loop, persistent (snapshot-restartable)
context memory, and test-time compute scaling. Backend: OpenRouter (model-agnostic).
Grounding: arXiv by default, pluggable web search. Interface: CLI + structured run logs.

## Fidelity policy
Prompts and constants are tagged by provenance: **verbatim** from the paper's supplementary
materials (SN8 pseudocode / SN9 prompts) where published, and **reconstructed in-style**
(clearly labeled) where the paper does not publish them. The paper's source code was not
released; SN8 + SN9 are the authors' stated reproducibility aids and our ground truth.
See the design spec for the full policy and the verbatim-vs-reconstructed breakdown.
