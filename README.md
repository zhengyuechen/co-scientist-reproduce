# Co-Scientist Reproduction

A faithful, runnable reproduction of the multi-agent architecture in
Gottweis, Weng, Daryin, Tu, et al., *"Accelerating scientific discovery with Co-Scientist,"*
**Nature** (2026), s41586-026-10644-y.

Given a research goal, the system generates research hypotheses, grounds and reviews them
(with a safety check), ranks them through an Elo tournament of LLM "scientific debates,"
evolves the strongest ones, and synthesizes a final research overview — all driven by a
Supervisor that chains the agents and improves quality as it spends more compute.

## How it works

A Supervisor seeds a global task queue and coordinates six specialized agents:

- **Generation** — proposes hypotheses (via literature-grounded reasoning and self-play scientific debate).
- **Reflection** — reviews each hypothesis for novelty and correctness and renders a safety verdict; hypotheses flagged unsafe are quarantined out of the tournament.
- **Ranking** — an Elo tournament in which pairs of hypotheses are compared through multi-turn LLM scientific debate (single-turn for lower-ranked pairs).
- **Evolution** — combines and refines the top hypotheses into stronger ones.
- **Proximity** — builds an embedding-similarity graph over the hypotheses.
- **Meta-review** — synthesizes system-wide feedback (fed back into later prompts) and the final research overview.

Agents trigger one another through the task queue: a new hypothesis is reviewed, a reviewed
hypothesis enters the tournament, the tournament runs continuously, and the strongest ideas
are evolved. Quality scales with the compute budget — more budget means more tournament and
evolution rounds, which raises the top hypotheses' Elo (the paper's test-time-compute-scaling
effect). The run loop is sequential, faithful to the supplement's SN8 `StartCoScientist`
pseudocode.

## Fidelity

Prompts and constants are tagged by provenance. Where the paper's supplement publishes a
prompt (SN9) or pseudocode (SN8), it is reproduced **verbatim** in `cosci/prompts/verbatim.py`
— byte-faithful, including the supplement's own irregular placeholder spellings. Where the
paper describes a behavior in its Methods but does not publish the prompt, it is
**reconstructed** in the same style and clearly labeled in `cosci/prompts/reconstructed.py`,
each constant tagged with the Methods/SN8/SN10 source it recreates. The paper's source code
was not released; SN8 and SN9 are the authors' stated reproducibility aids and the ground
truth here.

This reproduces the system's *architecture and behavior*, not Google's infrastructure: it is
model-agnostic (OpenRouter), grounds on arXiv by default (with a pluggable web-search backend),
and runs locally.

## Layout

- `cosci/` — typed config, domain models, Elo math, an async task queue, the OpenRouter client
  (plus a deterministic fake LLM for tests), snapshot-restartable context memory, the six
  `agents/`, the `prompts/`, the `supervisor` and `engine`, and `tools/` (literature grounding).
- `tests/` — unit and integration tests that run entirely against the deterministic fake LLM
  (no network).

## Running

Drive the engine in code (needs `OPENROUTER_API_KEY`):

```python
import asyncio
from cosci.config import load_config
from cosci.memory import ContextMemory
from cosci.engine import run_engine
from cosci.models import AgentName
from cosci.llm import OpenRouterClient
from cosci.tools.web_search import build_backend
from cosci.agents import (GenerationAgent, ReflectionAgent, RankingAgent,
                          EvolutionAgent, ProximityAgent, MetaReviewAgent)

cfg = load_config("config.yaml")
llm = OpenRouterClient(cfg)
grounding = build_backend(cfg)          # arXiv by default
agents = {
    AgentName.GENERATION: GenerationAgent(grounding=grounding),
    AgentName.REFLECTION: ReflectionAgent(grounding=grounding),
    AgentName.RANKING: RankingAgent(),
    AgentName.EVOLUTION: EvolutionAgent(),
    AgentName.PROXIMITY: ProximityAgent(),
    AgentName.META_REVIEW: MetaReviewAgent(),
}
overview = asyncio.run(run_engine("Your research goal here", ContextMemory(), llm, cfg, agents))
print(overview)
```

Run the tests (deterministic, no network) from the repository root:

```
conda run -n cosci-reproduce pytest -q
```
