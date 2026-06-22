import pytest
from cosci.llm import extract_json
from tests.fake_llm import FakeLLM

def test_extract_json_strips_fences():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('{"b": 2}') == {"b": 2}

def test_extract_json_raises_on_garbage():
    with pytest.raises(ValueError):
        extract_json("not json at all")

@pytest.mark.asyncio
async def test_fake_llm_is_deterministic_and_records():
    fake = FakeLLM(router=lambda agent, messages: f"reply-to-{agent}")
    out = await fake.complete("generation", [{"role": "user", "content": "hi"}])
    assert out == "reply-to-generation"
    assert fake.calls[0]["agent"] == "generation"
