from cosci.agents.text_utils import clean_title, split_atomic_hypotheses

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
