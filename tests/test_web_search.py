import pytest
import sys
import types
from cosci.tools.web_search import (Article, ArxivBackend, format_articles,
                                     build_backend, is_faithful_grounding,
                                     safe_search)
from cosci.config import load_config

def test_format_articles_empty_and_nonempty():
    assert format_articles([]) == ""
    out = format_articles([Article(title="T", summary="S", url="u", published="2024")])
    assert "T" in out and "S" in out and "u" in out and "2024" in out
    assert "[A1]" in out and "cite claims as [A1]" in out

class _FakeResult:
    def __init__(self, t): self.title=t; self.summary="abs "+t; self.entry_id="http://x/"+t; self.published="2024-01-01"
class _FakeClient:
    def results(self, search): return [_FakeResult("a"), _FakeResult("b")]

@pytest.mark.asyncio
async def test_arxiv_backend_parses_injected_client():
    arts = await ArxivBackend(client=_FakeClient()).search("quantum", max_results=2)
    assert [a.title for a in arts] == ["a", "b"]
    assert arts[0].summary == "abs a" and arts[0].url == "http://x/a"

@pytest.mark.asyncio
async def test_arxiv_backend_uses_small_page_size(monkeypatch):
    created = []

    class FakeSearch:
        def __init__(self, query, max_results):
            self.query = query
            self.max_results = max_results

    class FakeClient:
        def __init__(self, page_size=100):
            created.append(page_size)

        def results(self, search):
            assert search.query == "quantum"
            assert search.max_results == 3
            return [_FakeResult("a")]

    monkeypatch.setitem(
        sys.modules,
        "arxiv",
        types.SimpleNamespace(Client=FakeClient, Search=FakeSearch),
    )

    arts = await ArxivBackend().search("quantum", max_results=3)
    assert created == [3]
    assert [a.title for a in arts] == ["a"]

@pytest.mark.asyncio
async def test_safe_search_degrades_on_lookup_error():
    class FailingBackend:
        async def search(self, query, max_results=5):
            raise RuntimeError("Page request resulted in HTTP 429")

    assert await safe_search(FailingBackend(), "quantum") == ""

def test_build_backend_and_faithful_flag():
    cfg = load_config("config.yaml")  # default grounding.backend == "arxiv"
    assert isinstance(build_backend(cfg), ArxivBackend)
    assert is_faithful_grounding(cfg) is False
    cfg.grounding.backend = "none"
    assert build_backend(cfg) is None
    cfg.grounding.backend = "tavily"
    assert is_faithful_grounding(cfg) is True
    # unsupported backends must raise, not silently fall through to ArxivBackend
    cfg.grounding.backend = "serper"
    with pytest.raises(ValueError):
        build_backend(cfg)
