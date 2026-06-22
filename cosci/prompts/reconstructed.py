"""Reconstructed SN9-gap prompts.

Every prompt in this module is RECONSTRUCTED — it is NOT taken from the
supplement (SM). The Co-Scientist paper (Gottweis et al., Nature 2026)
describes these agent strategies in its Methods but did not publish the
corresponding prompts in SN9. Each constant below is authored from scratch
to fill that gap, written in the SN9 style of the verbatim prompts in
``cosci/prompts/verbatim.py`` (expert-role opening, an explicit ``Goal:``,
a ``Criteria``/``Preferences`` section where applicable, numbered
instructions, and an explicit closing output instruction). Verdict-style
prompts end with a required concluding line.

No literal JSON braces appear anywhere; every ``{...}`` token is a render
placeholder. Any structured/JSON output is described in words.
"""

# RECONSTRUCTED — not from SM. Methods intent: Generate a hypothesis by identifying testable intermediate assumptions via conditional reasoning hops, then aggregating.
# Recreates: Methods "iterative assumptions identification" Generation strategy (Methods p.25). No prompt published in SN9.
GEN_ITERATIVE_ASSUMPTIONS = """You are an expert tasked with formulating a novel and robust hypothesis by reasoning through a chain of testable intermediate assumptions.
Rather than proposing a final answer directly, build toward it through a sequence of conditional reasoning hops, then aggregate the surviving assumptions into a single coherent hypothesis.
This description is intended for an audience of domain experts.

Goal: {goal}

Criteria for a strong hypothesis:
{preferences}

Additional instructions:
{instructions}

Instructions:
1. Decompose the objective into a small set of intermediate assumptions that, if true, would jointly support a solution.
2. For each assumption, reason conditionally: state what would have to hold for it to be true, and what observable consequence would follow if it were true.
3. Assess the plausibility of each assumption in turn, discarding those that are weak, untestable, or unsupported, and noting the dependencies between assumptions.
4. Aggregate the surviving, mutually consistent assumptions into one detailed hypothesis, specifying the entities, mechanisms, and anticipated outcomes.
5. Make explicit which intermediate assumptions the final hypothesis depends on, so each can be independently tested.

Proposed hypothesis (detailed description for domain experts, with its supporting chain of assumptions):"""

# RECONSTRUCTED — not from SM. Methods intent: Propose hypotheses in under-explored areas, informed by existing hypotheses + meta-review overview.
# Recreates: Methods "research expansion" Generation strategy (Methods p.25). No prompt published in SN9.
GEN_RESEARCH_EXPANSION = """You are an expert tasked with proposing novel hypotheses that expand into under-explored regions of the research space for the following objective.
You are given an overview of the research directions already pursued by the system. Your task is to deliberately move beyond them, proposing hypotheses that open promising but neglected avenues rather than restating established ones.
This description is intended for an audience of domain experts.

Goal: {goal}

Criteria for a strong hypothesis:
{preferences}

Research overview of directions explored so far:
{research_overview}

Instructions:
1. Analyze the research overview to identify which areas, mechanisms, and assumptions have already been heavily explored.
2. Identify gaps: plausible directions, mechanisms, or analogies that are under-represented or absent from the existing work.
3. Select one or more of the most promising under-explored directions, briefly justifying why each is worth pursuing.
4. Develop a detailed, original hypothesis for the chosen direction, specifying the entities, mechanisms, and anticipated outcomes, and contrasting it with the directions already covered.

Proposed hypothesis (detailed description for domain experts, situated relative to the existing research overview):"""

# RECONSTRUCTED — not from SM. Methods intent: Fast tool-free review: correctness, quality, novelty, safety; flag clearly-flawed/unsafe for discard.
# Recreates: Methods "initial review" (no external tools, fast discard; Methods pp.26-27). No prompt published in SN9.
REFLECT_INITIAL = """You are an expert in scientific hypothesis evaluation performing a rapid initial review without consulting any external tools or literature.
Your task is to judge, using only your own expert knowledge, whether the hypothesis is worth advancing or should be discarded as clearly flawed or unsafe.

Goal: {goal}

Hypothesis:
{hypothesis}

Instructions:
1. Correctness: assess whether the hypothesis is logically sound and consistent with well-established scientific knowledge. Note any clear errors.
2. Quality: assess whether it is sufficiently specific, detailed, and well-formed to be useful.
3. Novelty: assess whether it appears to offer something beyond obvious or already-established explanations.
4. Safety: assess whether pursuing the hypothesis could be unethical or unsafe.
5. Decision: recommend whether the hypothesis should be discarded now (clearly flawed or unsafe) or allowed to proceed to fuller review.

Respond with concise reasoning for each criterion, then conclude with "decision: <discard or proceed>" and a brief reason."""

# RECONSTRUCTED — not from SM. Methods intent: Full review with literature grounding: novelty + correctness, return assessments + a safety verdict + references.
# Recreates: Methods "full review" w/ web-search grounding (Methods pp.26-27); SN8 ReviewHypothesis one-liner; SN10.5 example output. No prompt published in SN9.
REFLECT_FULL = """You are an expert in scientific hypothesis evaluation performing a full, literature-grounded review.
Your task is to evaluate the hypothesis for novelty and correctness, supporting your judgments with the provided literature, and to render a safety verdict.

Goal: {goal}

Hypothesis:
{hypothesis}

Literature review and analytical rationale (chronologically ordered, beginning with the most recent analysis):
{articles_with_reasoning}

Instructions:
1. Correctness: evaluate whether the hypothesis is logically sound and consistent with the provided literature. Cite the specific articles that support or contradict it.
2. Novelty: evaluate whether the hypothesis is novel relative to the provided literature, distinguishing genuinely new claims from those already established.
3. Grounding: for each major claim, identify the supporting or refuting evidence among the provided articles, and note where evidence is missing.
4. References: list the articles you relied on in reaching your assessment.
5. Safety: assess whether pursuing the hypothesis could be unethical or unsafe.

Provide your novelty and correctness assessments with supporting references, then conclude with "safety: <safe or unsafe>" and a brief reason."""

# RECONSTRUCTED — not from SM. Methods intent: Decompose hypothesis into core + sub-assumptions, evaluate each independently for plausibility.
# Recreates: Methods "deep verification review" (Methods pp.26-27); SN8 ReviewHypothesis "break down into core assumptions" one-liner; SN10.6/10.7 examples. No prompt published in SN9.
REFLECT_DEEP_VERIFICATION = """You are an expert in scientific hypothesis evaluation performing a deep verification review.
Your task is to decompose the hypothesis into its core assumption and its constituent sub-assumptions, then evaluate the plausibility of each one independently, so that a flaw in any single assumption can be localized.

Hypothesis:
{hypothesis}

Instructions:
1. Identify the single core assumption on which the hypothesis fundamentally depends.
2. Enumerate the sub-assumptions that must additionally hold for the hypothesis to be valid.
3. For each assumption and sub-assumption, evaluate its plausibility independently of the others, citing established knowledge and stating whether it is well-supported, uncertain, or likely false.
4. Identify which assumptions are most critical, such that their failure would invalidate the hypothesis.
5. Summarize how the assumption-level findings bear on the overall plausibility of the hypothesis.

Provide your per-assumption analysis, then conclude with "verification: <verified, uncertain, or invalidated>" and a brief reason."""

# RECONSTRUCTED — not from SM. Methods intent: Step-wise simulate the hypothesis's mechanism/experiment to surface failure modes.
# Recreates: Methods "simulation review" (Methods pp.26-27). No prompt published in SN9.
REFLECT_SIMULATION = """You are an expert in scientific hypothesis evaluation performing a review by simulation.
Your task is to mentally simulate, step by step, the mechanism or experiment implied by the hypothesis, in order to surface the points at which it is most likely to fail.

Hypothesis:
{hypothesis}

Instructions:
1. Restate the mechanism or experiment that the hypothesis predicts, identifying its starting conditions and expected end state.
2. Walk through the process one step at a time, describing what should happen at each step if the hypothesis holds.
3. At each step, identify potential failure modes: where the predicted behavior could break down, give an unexpected result, or be confounded.
4. Note which steps are the most fragile and what observations would reveal a failure at each.
5. Summarize the overall robustness of the hypothesis in light of the simulated process and the failure modes identified.

Provide your step-by-step simulation and identified failure modes, then conclude with "simulation: <robust, fragile, or fails>" and a brief reason."""

# RECONSTRUCTED — not from SM. Methods intent: Re-review adapting to tournament results + meta-review feedback.
# Recreates: Methods "recurrent/tournament review" — adapts reviews using tournament results + meta-review feedback (Methods pp.26-27). No prompt published in SN9.
REFLECT_RECURRENT = """You are an expert in scientific hypothesis evaluation performing a recurrent re-review of a hypothesis that has already been through earlier review and tournament rounds.
Your task is to update your assessment in light of how the hypothesis has fared in the tournament and of the meta-review feedback gathered across the system.

Hypothesis:
{hypothesis}

Tournament results and meta-review feedback:
{notes}

Instructions:
1. Summarize what the tournament results and meta-review feedback reveal about this hypothesis's standing relative to its competitors.
2. Identify the recurring strengths and weaknesses that the feedback attributes to this hypothesis or to hypotheses like it.
3. Re-evaluate the hypothesis's correctness, novelty, and quality, explicitly updating any earlier judgment that the new evidence changes.
4. Recommend concrete revisions that would address the weaknesses surfaced by the feedback.

Provide your updated review, then conclude with "assessment: <improved, unchanged, or weakened>" and a brief reason."""

# RECONSTRUCTED — not from SM. Methods intent: Combine the best parts of two top hypotheses into a stronger one.
# Recreates: SN8 EvolveTopHypotheses "Combine the best parts of [H1] and [H2] into a new, stronger hypothesis"; Methods "combination" Evolution strategy (p.27).
EVO_COMBINE = """You are an expert researcher tasked with combining the strongest elements of two top-performing hypotheses into a single, stronger hypothesis.
Your task is not to simply concatenate them, but to synthesize their best parts into a coherent whole that retains novelty, logical consistency, and specificity.

Goal: {goal}

Criteria for a strong hypothesis:
{preferences}

Hypotheses to combine:
{hypotheses}

Instructions:
1. Provide a concise introduction to the relevant scientific domain shared by the hypotheses.
2. For each hypothesis, identify its most valuable elements, including the mechanisms or insights that give it strength.
3. Identify where the two are complementary and where they conflict, and decide how to reconcile any conflicts.
4. CORE CONTRIBUTION: develop a single detailed hypothesis that integrates the strongest complementary elements, specifying the entities, mechanisms, and anticipated outcomes.
5. Explain how the combined hypothesis improves on each of the originals against the stated criteria.

Response:"""

# RECONSTRUCTED — not from SM. Methods intent: Refine a hypothesis to be simpler and more testable.
# Recreates: SN8 EvolveTopHypotheses "Refine [H3] to make it simpler and more testable"; Methods "simplification" Evolution strategy (p.27).
EVO_SIMPLIFY = """You are an expert in scientific research tasked with refining a hypothesis to make it simpler and more directly testable, while preserving its novelty and explanatory power.

Goal: {goal}

Guidelines:
1. Begin with a concise restatement of the hypothesis as you understand it.
2. Identify sources of unnecessary complexity: superfluous assumptions, vague terms, or untestable commitments.
3. Refine the hypothesis to its simplest form that still addresses the goal, removing or tightening each source of complexity.
4. CORE CONTRIBUTION: state the simplified hypothesis and specify a concrete, feasible way to test it, including what observation would confirm or refute it.
5. Confirm that the simplification preserves the hypothesis's essential novelty and explanatory content.

Evaluation Criteria:
{preferences}

Original hypothesis:
{hypothesis}

Response:"""

# RECONSTRUCTED — not from SM. Methods intent: Strengthen a hypothesis by grounding it in retrieved literature.
# Recreates: Methods "enhancement through grounding" Evolution strategy (p.27). No prompt published in SN9.
EVO_GROUNDING = """You are an expert in scientific research tasked with strengthening a hypothesis by grounding it in the retrieved literature, enhancing its support and specificity while preserving its novelty and logical coherence.

Goal: {goal}

Guidelines:
1. Begin with a concise overview of the relevant scientific domain.
2. Review the retrieved literature and identify findings that support, qualify, or challenge the hypothesis.
3. Revise the hypothesis so that each major claim is anchored to the supporting evidence, and adjust or hedge any claim that the literature does not support.
4. CORE CONTRIBUTION: present the strengthened, literature-grounded hypothesis in detail, specifying the entities, mechanisms, and anticipated outcomes, and cite the articles that support each key claim.
5. Note any remaining gaps where evidence is missing and that future work would need to address.

Evaluation Criteria:
{preferences}

Original hypothesis:
{hypothesis}

Literature review and analytical rationale (chronologically ordered, beginning with the most recent analysis):
{articles_with_reasoning}

Response:"""

# RECONSTRUCTED — not from SM. Methods intent: Generate a new hypothesis inspired by (not copied from) existing ones.
# Recreates: Methods "inspiration from existing hypotheses" Evolution strategy (p.27); close cousin of the verbatim EVO_OUT_OF_BOX (SN9.4 #2).
EVO_INSPIRATION = """You are an expert researcher tasked with generating a novel, singular hypothesis inspired by analogous elements drawn from existing hypotheses.
The new hypothesis should be inspired by, not copied from, the provided ideas. Think out-of-the-box.

Goal: {goal}

Instructions:
1. Provide a concise introduction to the relevant scientific domain.
2. Examine the provided hypotheses and identify the underlying principles, analogies, or patterns that make them promising.
3. Abstract away from their specific content to the transferable ideas that could seed a genuinely new direction.
4. CORE HYPOTHESIS: develop a single detailed and original hypothesis that applies these analogous principles in a new way to achieve the goal. This should not be a mere aggregation of the provided ideas.

Criteria for a robust hypothesis:
{preferences}

Inspiration may be drawn from the following hypotheses (utilize analogy and inspiration, not direct replication):
{hypotheses}

Response:"""

# RECONSTRUCTED — not from SM. Methods intent: Synthesize top hypotheses into a coherent research overview: directions + justifications.
# Recreates: SN8 GenerateFinalResearchOverview "Synthesize these top-ranked hypotheses into a single, coherent research overview"; Methods Meta-review "research overview".
META_RESEARCH_OVERVIEW = """You are an expert in scientific research and synthesis.
Your task is to synthesize the top-ranked hypotheses into a coherent research overview that organizes them into distinct research directions and justifies each.

Goal: {goal}

Top-ranked hypotheses:
{hypotheses}

Instructions:
1. Identify the major themes shared across the top hypotheses and group them into a small number of distinct, well-defined research directions.
2. For each research direction, summarize its central idea and the hypotheses that support it.
3. For each direction, provide a justification: why it is promising for the stated goal, and what evidence or reasoning supports pursuing it.
4. Note the relationships between directions, including where they are complementary and where they compete.
5. Conclude with a prioritized summary indicating which directions appear most promising and why.

Produce a structured research overview covering the research directions, their supporting hypotheses, and their justifications.

Response:"""

# RECONSTRUCTED — not from SM. Methods intent: Analyze all reviews + debate transcripts → common strengths/weaknesses as system-wide feedback.
# Recreates: SN8 GenerateSystemFeedback "Analyze all these critiques... summarize as feedback for the whole system"; Methods Meta-review feedback loop (p.26, appended to agent prompts next iteration).
META_SYSTEM_FEEDBACK = """You are an expert in scientific research and meta-analysis.
Your task is to analyze the full set of reviews and debate transcripts produced by the system and distill them into system-wide feedback: the recurring strengths and weaknesses that should guide future hypothesis generation and review.

Goal: {goal}

Provided reviews and debate transcripts for meta-analysis:
{reviews}

Instructions:
1. Read across all of the provided reviews and transcripts, focusing on patterns rather than any single proposal.
2. Identify the strengths that recur across well-regarded hypotheses.
3. Identify the weaknesses, critiques, and failure modes that recur across poorly-regarded hypotheses.
4. Translate these patterns into actionable, system-wide guidance for generating and reviewing future hypotheses.
5. Refrain from evaluating any individual proposal or review; focus on the synthesized, system-level feedback.

Produce a structured meta-analysis report covering the common strengths, common weaknesses, and the resulting actionable guidance.

Response:"""

# RECONSTRUCTED — not from SM. Methods intent: Parse a natural-language research goal into preferences, attributes, constraints; return a JSON object (described in words).
# Recreates: Methods "research goal -> ResearchPlan configuration" (Methods p.23); SN10.1 example (Preferences/Attributes/Constraints). No prompt published in SN9.
SUP_GOAL_PARSE = """You are an expert at interpreting natural-language research goals and converting them into a structured research plan configuration.
Your task is to read the raw research goal and extract its preferences, attributes, and constraints.

Raw research goal:
{goal_raw}

Instructions:
1. Restate the core objective of the research goal in a single clear sentence.
2. Extract the preferences: the criteria the user would use to judge a strong hypothesis (such as novelty, feasibility, or specificity).
3. Extract the attributes: the desired qualities or focus of the hypotheses to be generated.
4. Extract the constraints: any explicit limits, exclusions, ethical boundaries, or required conditions.
5. Where the goal is silent, mark the corresponding field as unspecified rather than inventing requirements.

Return the result as a JSON object with a key for the restated goal, a key for preferences, a key for attributes, and a key for constraints, where preferences, attributes, and constraints are each lists of strings.

Response:"""

# RECONSTRUCTED — not from SM. Methods intent: Safety-review a research goal; conclude safe/unsafe with reason.
# Recreates: SN7 safety — goal-level safety review run before any generation (unsafe goal aborts the run). No prompt published in SN9.
SUP_SAFETY_REVIEW_GOAL = """You are an expert in research safety and ethics tasked with reviewing a proposed research goal before any hypotheses are generated for it.
Your task is to determine whether the goal can be pursued safely and ethically, or whether it could enable harm.

Raw research goal:
{goal_raw}

Instructions:
1. Restate the research goal and identify its apparent intent.
2. Consider whether the goal, or the most likely lines of inquiry it would invite, could cause harm or enable dangerous, unethical, or illegal activity.
3. Consider dual-use concerns: legitimate framings that could nonetheless be redirected toward harm.
4. Weigh the plausible benefits against the plausible risks, and note any conditions under which the goal would be acceptable.

Provide your reasoning, then conclude with "safety: <safe or unsafe>" and a brief reason."""

# RECONSTRUCTED — not from SM. Methods intent: Safety-review a hypothesis; conclude safe/unsafe with reason.
# Recreates: SN7 safety — per-hypothesis safety review. Per the supplement, hypotheses flagged unsafe are EXCLUDED from the tournament (do not enter ranking/evolution/overview); enforced in the Supervisor (Phase 4). No prompt published in SN9.
SUP_SAFETY_REVIEW_HYPOTHESIS = """You are an expert in research safety and ethics tasked with reviewing a generated hypothesis in the context of its research goal.
Your task is to determine whether the hypothesis can be pursued safely and ethically, or whether it could enable harm.

Goal: {goal}

Hypothesis:
{hypothesis}

Instructions:
1. Summarize what the hypothesis proposes and how it would be investigated.
2. Consider whether pursuing or publishing the hypothesis could cause harm or enable dangerous, unethical, or illegal activity.
3. Consider dual-use concerns: whether otherwise legitimate aspects of the hypothesis could be redirected toward harm.
4. Weigh the plausible scientific value against the plausible risks, and note any conditions under which the hypothesis would be acceptable.

Provide your reasoning, then conclude with "safety: <safe or unsafe>" and a brief reason."""
