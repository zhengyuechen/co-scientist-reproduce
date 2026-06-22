import re
from cosci.prompts import reconstructed as R

REQUIRED = {
    "GEN_ITERATIVE_ASSUMPTIONS": ["{goal}", "{preferences}", "{instructions}"],
    "GEN_RESEARCH_EXPANSION": ["{goal}", "{preferences}", "{research_overview}"],
    "REFLECT_INITIAL": ["{goal}", "{hypothesis}"],
    "REFLECT_FULL": ["{goal}", "{hypothesis}", "{articles_with_reasoning}"],
    "REFLECT_DEEP_VERIFICATION": ["{hypothesis}"],
    "REFLECT_SIMULATION": ["{hypothesis}"],
    "REFLECT_RECURRENT": ["{hypothesis}", "{notes}"],
    "EVO_COMBINE": ["{goal}", "{preferences}", "{hypotheses}"],
    "EVO_SIMPLIFY": ["{goal}", "{preferences}", "{hypothesis}"],
    "EVO_GROUNDING": ["{goal}", "{preferences}", "{hypothesis}", "{articles_with_reasoning}"],
    "EVO_INSPIRATION": ["{goal}", "{preferences}", "{hypotheses}"],
    "META_RESEARCH_OVERVIEW": ["{goal}", "{hypotheses}"],
    "META_SYSTEM_FEEDBACK": ["{goal}", "{reviews}"],
    "SUP_GOAL_PARSE": ["{goal_raw}"],
    "SUP_SAFETY_REVIEW_GOAL": ["{goal_raw}"],
    "SUP_SAFETY_REVIEW_HYPOTHESIS": ["{goal}", "{hypothesis}"],
}


def test_all_constants_present_nonempty():
    for name in REQUIRED:
        assert isinstance(getattr(R, name), str) and len(getattr(R, name)) > 80


def test_required_placeholders_present():
    for name, phs in REQUIRED.items():
        body = getattr(R, name)
        for ph in phs:
            assert ph in body, f"{name} missing {ph}"


def test_no_literal_json_braces():
    # the only legal {..} tokens are known placeholders; no literal JSON braces
    known = {ph.strip("{}") for phs in REQUIRED.values() for ph in phs}
    for name in REQUIRED:
        for tok in re.findall(r"\{([^{}]*)\}", getattr(R, name)):
            assert tok in known, f"{name} has unexpected brace token: {{{tok}}}"
