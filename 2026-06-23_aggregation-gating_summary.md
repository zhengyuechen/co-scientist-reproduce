# Tier 0: capture and use the criticism the model already writes

**Date:** 2026-06-23
**Branch:** `tier0-parse-and-gate`
**Run this explains:** `results/2026-06-22_223642_is-wavefunction-collapse-physical/` (the "Is wavefunction collapse physical?" overview)

## The problem in one sentence

The system was already writing the right criticism — it named Penrose, Diósi, GRW, and CSL
dozens of times and even called hypotheses "a synthesis of existing concepts" — but that
criticism was trapped in unread review text. Nothing read it, so nothing acted on it, and a
polished "three promising directions" report came out the other end anyway.

## Why that happened (the gates)

Only two model judgments had any mechanical effect in the whole system:

1. **the safety flag** — unsafe hypotheses get filtered out, and
2. **the tournament winner** — which drives the Elo ranking.

Everything else the model "decided" was free text. In particular:

- Each review has a `scores` field meant to hold numeric judgments. **It was empty (`{}`) in
  all twelve reviews of that run.** The novelty assessment the model wrote in prose was never
  turned into a number.
- The deep-verification verdict (`verification: verified / uncertain / invalidated`) was never
  read.
- The final overview was built from hypothesis text + tournament rank, and **threw away the
  review bodies** — including every "this resembles Diósi–Penrose" sentence.

So the cleanest restatement of an existing model (G1) actually *won* the tournament, and the
overview presented it as a promising new direction.

## What this change does

It captures the verdicts the model already produces and lets them affect the outcome. No new
model, no search backend, no smarter prompt reasoning required.

1. **Parse the verdicts** (`reflection.py`). After each review we now read:
   - a `novelty: <1–10>` score from the full review (1 = a restatement of an existing model,
     10 = no comparable prior art), and
   - the `verification: <verified|uncertain|invalidated>` verdict from the deep review.
   These go into `review.scores` (no longer empty) and onto the hypothesis itself.

2. **Force the comparison** (`REFLECT_FULL` prompt). The full review must now name the single
   closest existing model and say whether the hypothesis is the same thing under a relabeling of
   terms — and if so, score novelty low. New terminology no longer counts as new content.

3. **Gate the overview** (`meta_review.py`). Before the final report is written, hypotheses whose
   own reviews condemned them — novelty below a threshold (default 5/10) or verification
   `invalidated` — are excluded. If nothing clears the bar, the overview is told to say plainly
   that the area is well-trodden and no novel direction emerged, instead of inventing one.

   The gate **fails open**: a hypothesis with no parsed score is kept, so this never
   retroactively punishes missing data (e.g. older runs).

## What it does and doesn't fix

- **Does:** strips confident restatements out of the final report. On this prompt, a correctly
  working system should now produce a *less* impressive, more skeptical overview — which is the
  right answer, because the six hypotheses really are existing collapse models relabeled.
- **Doesn't merge duplicates:** novelty-gating craters the two "information-saturation" twins
  (G1 and G2) *equally*, but it won't combine them into one. Merging duplicates needs the
  Proximity agent to actually be read by something — it currently writes a similarity graph that
  nobody consumes. That's a separate, later fix.
- **Grounding untouched:** the one grounded review in 223642 retrieved dark-energy *telescope*
  papers for "Topological Snap" and wrongly boosted novelty. Grounding is a later tier and, on a
  well-known topic like this, lower priority than it looks — the model recalls the prior art
  without it.

## The test to run next (your prediction, now runnable)

Re-run "Is wavefunction collapse physical?" on the real model. Prediction: the restatement
hypotheses (the information-saturation and metric-solidification ones) get low novelty scores,
get dropped, and the confident three-direction conclusion does not survive. If novelty *doesn't*
crater even after it's parsed and gated, the problem is in the model's equivalence reasoning, not
the pipeline — and that is the outcome worth knowing.

## Verification done so far

- 5 new tests (`tests/test_aggregation_gating.py`): novelty parsing/clamping, the gate predicate,
  the overview excluding a low-novelty hypothesis *even when it won the tournament*, and the
  all-restatements case. Full suite: 102 passing.
- End-to-end (offline): through the engine, review `scores` now read
  `[{'novelty': 2.0}, {'verification': 0.5}]` instead of `{}`, and reviewed restatements are
  excluded from the overview.

## Files changed

- `cosci/agents/reflection.py` — parse novelty + verification, populate `scores` and the hypothesis.
- `cosci/agents/meta_review.py` — `passes_novelty_gate` + filter the overview, with an honest note.
- `cosci/prompts/reconstructed.py` — `REFLECT_FULL` forces the closest-model comparison + a parseable `novelty:` line.
- `cosci/models.py` — `Hypothesis.novelty`, `Hypothesis.verification`.
- `cosci/config.py` — `overview.min_novelty` (default 5.0).
- `tests/test_aggregation_gating.py` — new.
