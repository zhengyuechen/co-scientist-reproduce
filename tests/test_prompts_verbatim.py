from cosci.prompts import verbatim as V

def test_all_eight_prompts_present_and_nonempty():
    for name in ["GEN_LITERATURE", "GEN_DEBATE", "REFLECT_OBSERVATION",
                 "RANK_SINGLE_TURN", "RANK_MULTI_TURN", "EVO_FEASIBILITY",
                 "EVO_OUT_OF_BOX", "META_SYNTHESIS"]:
        assert isinstance(getattr(V, name), str) and len(getattr(V, name)) > 100

def test_key_placeholders_present():
    assert "{goal}" in V.GEN_LITERATURE
    assert "{articles_with_reasoning}" in V.GEN_LITERATURE
    assert "{idea_attributes}" in V.GEN_DEBATE and "{transcript}" in V.GEN_DEBATE
    assert "{article}" in V.REFLECT_OBSERVATION and "{hypothesis}" in V.REFLECT_OBSERVATION
    # supplement's inconsistent spacing preserved verbatim:
    assert "{hypothesis 1}" in V.RANK_SINGLE_TURN and "{review 1}" in V.RANK_SINGLE_TURN
    assert "{review1}" in V.RANK_MULTI_TURN and "{review 2}" in V.RANK_MULTI_TURN
    assert "{hypotheses}" in V.EVO_OUT_OF_BOX
    assert "{reviews}" in V.META_SYNTHESIS

def test_placeholder_map_covers_space_names():
    assert V.PLACEHOLDER_MAP["hypothesis 1"] == "hypothesis_1"
    assert V.PLACEHOLDER_MAP["review1"] == "review_1"
    assert V.PLACEHOLDER_MAP["review 2"] == "review_2"
