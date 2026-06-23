from cosci.agents.text_utils import clean_title, is_scaffolding, split_atomic_hypotheses

# Real degenerate output (G6) and real preamble-but-valid hypotheses (G1, G5) from the
# 2026-06-23 falsify run, abbreviated.
G6_SCAFFOLD = ("Since the research overview provided in the prompt was blank, I have reconstructed "
               "the standard landscape of foundational research regarding the physicality of "
               "wavefunction collapse. ### Analysis of the Research Landscape **Heavily Explored "
               "Directions:** 1. The Epistemic View (Copenhagen, QBism, Relational)")
G1_REAL = ("Based on the objective of determining whether wavefunction collapse is a physical "
           "process, and synthesizing the current discourse, I propose the following hypothesis.\n\n"
           "### Proposed Hypothesis: The Spacetime-Metric Instability (SMI) Hypothesis\n\n"
           "When a massive object exists in spatial superposition...")
G5_REAL = ("### 1. Intermediate Assumptions and Conditional Reasoning\n\nTo determine if wavefunction "
           "collapse is physical, we must first establish that unitary evolution is insufficient.\n\n"
           "**Assumption 1:** The incompatibility of superposition and measurement.")


def test_is_scaffolding_rejects_task_meta_and_outline_leaks():
    assert is_scaffolding(G6_SCAFFOLD) is True                 # task meta-commentary / survey
    assert is_scaffolding(G5_REAL) is True                     # outline header used in place of a hypothesis
    assert is_scaffolding("1. Analysis of Existing Research Directions\n...") is True


def test_is_scaffolding_keeps_real_hypotheses_with_preamble():
    assert is_scaffolding(G1_REAL) is False     # real hypothesis, just opens with a preamble
    assert is_scaffolding("Wavefunction collapse is a thermodynamic phase transition.") is False
    assert is_scaffolding("1. The collapse occurs when entropy exceeds a threshold.") is False  # numbered, but a claim


def test_clean_title_skips_based_on_preamble_to_real_title():
    assert clean_title(G1_REAL) == "The Spacetime-Metric Instability (SMI) Hypothesis"

# Real bundled debate output (abbreviated) that previously became ONE hypothesis
# titled "I'll initiate this collaborative discourse...".
DEBATE_BUNDLE = """I'll initiate this collaborative discourse by proposing three distinct novel hypotheses regarding whether wavefunction collapse is physical.

---

**Hypothesis 1: Thermodynamic Decoherence Boundary (TDB) Hypothesis**

Wavefunction collapse is a physical process that occurs at a specific thermodynamic threshold, with entropy production exceeding a critical value.

**Hypothesis 2: Stochastic Gravitational Decoherence (SGD) Hypothesis**

Collapse is driven by gravitational self-interaction of the mass distribution, following a Diosi-Penrose timescale.

**Hypothesis 3: Observer Perceptual Horizon (OPH) Hypothesis**

Collapse is a relational boundary condition enforced at the observer's macroscopic perceptual state.
"""


def test_clean_title_strips_label_and_markdown():
    # Real literature-review first line.
    title = clean_title("**Proposed Hypothesis: The Relational Stochastic Boundary Hypothesis of Effective Collapse**\n\nBody...")
    assert title == "The Relational Stochastic Boundary Hypothesis of Effective Collapse"


def test_clean_title_skips_conversational_preamble():
    title = clean_title(DEBATE_BUNDLE)
    assert title == "Thermodynamic Decoherence Boundary (TDB) Hypothesis"
    assert "I'll initiate" not in title
    assert "*" not in title


def test_clean_title_strips_hypothesis_number_label():
    assert clean_title("### Hypothesis 2: Gravitational Collapse") == "Gravitational Collapse"


def test_clean_title_preserves_a_plain_title():
    assert clean_title("Mechanism B drives the observed decoherence") == "Mechanism B drives the observed decoherence"


def test_clean_title_does_not_eat_keyword_in_running_title():
    # "Theory of ..." is a real title, not a "Theory:" label — must survive.
    assert clean_title("Theory of relational collapse") == "Theory of relational collapse"


def test_clean_title_truncates_long_line():
    t = clean_title("word " * 40)
    assert len(t) <= 81 and t.endswith("…")


def test_split_breaks_bundle_into_atomic_hypotheses():
    chunks = split_atomic_hypotheses(DEBATE_BUNDLE)
    assert len(chunks) == 3
    assert chunks[0].startswith("**Hypothesis 1:")
    assert "Hypothesis 2" in chunks[1]
    # the preamble before the first marker is dropped
    assert not any("I'll initiate" in c for c in chunks)


def test_split_keeps_single_hypothesis_whole():
    single = "**Proposed Hypothesis: Effective collapse**\n\n1. Core proposition\n2. Prediction\n3. Test"
    chunks = split_atomic_hypotheses(single)
    assert chunks == [single.strip()]


def test_split_does_not_split_internal_numbered_assumptions():
    body = ("The collapse is thermodynamic.\n\n"
            "Supporting assumptions:\n1. Entropy is monotone.\n2. Decoherence is fast.\n3. The bath is Markovian.")
    assert split_atomic_hypotheses(body) == [body.strip()]
