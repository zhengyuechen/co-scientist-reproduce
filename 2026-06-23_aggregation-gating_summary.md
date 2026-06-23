# Making the criticism count: parse, prune, and name the prior art

**Date:** 2026-06-23
**Branch:** `tier0-parse-and-gate` (kept unmerged)
**Runs:** `results/2026-06-22_223642_…` (the original laundered overview), `…_falsify_…` (Tier 0 only),
`…_falsify2_…` (the full pipeline / acceptance run)

## The problem (plain version)

The system was already writing the right criticism — it named Penrose, Diósi, GRW, CSL dozens of
times and called hypotheses "a synthesis of existing concepts" — but every skeptical judgment was
trapped in unread review text. The only two model judgments that did anything were the safety flag
and the tournament winner. So the cleanest restatement of an existing model could win the
tournament and be written up as a promising new direction. And the candidate pool was poisoned:
three of six "hypotheses" were scaffolding (the model talking about the task), and the novelty
scorer rated the worst of them the most novel.

## The four fixes (in the order that mattered)

1. **Generation hygiene — stop handing the gate garbage.** Generation now rejects chunks that are
   task meta-commentary / surveys / refusals (`is_scaffolding`) and regenerates a strategy once if
   it produced nothing usable. The title cleaner gained coverage for more preambles ("Based on…",
   "To determine…", "Since…") so a real hypothesis behind a preamble keeps its real title.

2. **Reliable verdicts — make every review emit a score.** `REFLECT_FULL` now ends in a strict
   required `novelty: <1-10>` line and floors non-hypotheses (surveys, outlines) to novelty 1. If
   the model still skips the line, reflection re-asks for just that one line. The parsed novelty and
   the deep-verification verdict now land in `review.scores` (previously empty on every review).

3. **A gate that actually prunes.** Before, the gate only hid low-novelty hypotheses from the final
   overview; they stayed active and kept competing. Now a hypothesis its own reviews condemn (novelty
   below threshold, default 5, or verification `invalidated`) is deactivated at reflection — before
   the tournament — so it stops surviving everywhere. It still fails *open* on a missing score.

4. **Tier 1 — condition the overview on the criticism.** Each surviving hypothesis carries its review
   into the synthesis, and the overview is instructed to name the single closest existing model per
   direction and flag restatements rather than launder them — reporting how many were pruned.

## Acceptance test (same prompt, "Is wavefunction collapse physical?", grounding off)

The test was: does a self-identified restatement get **pruned** (not just hidden), and does the
overview **name the prior art**? Both now pass.

| Signal | Before (Tier 0 only) | After (full pipeline) |
|---|---|---|
| Novelty scores parsed | 3 / 6 | 4 / 6 |
| Hypotheses pruned (active=False) | 0 (overview-hidden only) | **2** (G2 nov 2, G6 nov 4 — out of tournament + overview) |
| Overview names Penrose | 0× | **4×** (+ Bekenstein 2×) |
| Overview verdict | "promising directions" | **"this conceptual space is largely well-trodden"** |
| Directions flagged as restatements | none | **Directions 2 & 3 explicitly** |

The overview now contains a "Closest Existing Model" line per direction and concludes that the
attempts "rehash either thermodynamic decoherence or pre-geometric quantum gravity concepts." That
is the correct, deflationary answer for a well-trodden question — a *less* impressive artifact than
the original, which is the point.

## What's still imperfect (the model ceiling, now isolated)

The remaining lenience is the reflection model, not the pipeline. `reflection` runs on
`z-ai/glm-4.7-flash`, and:

- it rated the holographic/Bekenstein restatement (G1) **novelty 7**, so Direction 1 survived and the
  model defended it as "a novel synthesis" — even while now naming Penrose OR and the Bekenstein
  bound for it. The plumbing prunes whatever is scored low; the flash model just scores some
  restatements too high.
- 2 of 6 reviews still emitted no parseable novelty even after the re-ask (flash unreliability), and
  it fumbles the tournament's `better idea:` format too (the `parse_label returned None` warnings).

The pipeline is now correct and the laundering is fixed; the last increment of output quality is a
**reflection-model** question. The clean next experiment is to put `reflection` on `glm-5.2` and
re-run: prediction is that the holographic restatement also craters and the overview deflates
further. If even the strong model rates it novel with the comparison forced, that is a genuine
finding about how hard equivalence-under-relabeling is — not a pipeline bug.

## Files

- `cosci/agents/text_utils.py` — `is_scaffolding`; wider preamble coverage in `clean_title`.
- `cosci/agents/generation.py` — reject scaffolding chunks, regenerate once.
- `cosci/agents/reflection.py` — re-ask for a missing novelty line; prune (deactivate) on gate fail.
- `cosci/agents/base.py` — `passes_novelty_gate` (shared).
- `cosci/agents/meta_review.py` — carry reviews into the overview; report pruned count.
- `cosci/prompts/reconstructed.py` — strict `REFLECT_FULL` verdict + degenerate floor; Tier 1 overview prompt.
- `cosci/models.py` — `Hypothesis.novelty`, `.verification`, `.pruned_reason`; `OverviewCfg.min_novelty`.
- Tests: `test_text_utils.py`, `test_agent_generation.py`, `test_aggregation_gating.py` (110 passing).
