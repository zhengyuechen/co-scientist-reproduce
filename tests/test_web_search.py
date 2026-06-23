import pytest
from cosci.tools.web_search import (Article, ArxivBackend, format_articles,
                                     build_backend, is_faithful_grounding)
from cosci.config import load_config

def test_format_articles_empty_and_nonempty():
    assert format_articles([]) == ""
    out = format_articles([Article(title="T", summary="S", url="u", published="2024")])
    assert "T" in out and "S" in out and "u" in out and "2024" in out

class _FakeResult:
    def __init__(self, t): self.title=t; self.summary="abs "+t; self.entry_id="http://x/"+t; self.published="2024-01-01"
class _FakeClient:
    def results(self, search): return [_FakeResult("a"), _FakeResult("b")]

@pytest.mark.asyncio
async def test_arxiv_backend_parses_injected_client():
    arts = await ArxivBackend(client=_FakeClient()).search("quantum", max_results=2)
    assert [a.title for a in arts] == ["a", "b"]
    assert arts[0].summary == "abs a" and arts[0].url == "http://x/a"

def test_build_backend_and_faithful_flag():
    cfg = load_config("config.yaml")  # default grounding.backend == "arxiv"
    assert isinstance(build_backend(cfg), ArxivBackend)
    assert is_faithful_grounding(cfg) is False
    cfg.grounding.backend = "none"
    assert build_backend(cfg) is None
    cfg.grounding.backend = "tavily"
    assert is_faithful_grounding(cfg) is True
