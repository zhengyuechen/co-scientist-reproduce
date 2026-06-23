"""Pluggable literature-search layer for the Co-Scientist system.

NOTE: This backend layer is OURS — the paper used broad web search (Tavily / Google).
      arXiv-only is a deliberate fidelity compromise: free, no key required, but
      limited to preprints.
Backends: arXiv (default, free), none, or Tavily (web search, needs WEB_SEARCH_API_KEY)
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

log = logging.getLogger(__name__)


@dataclass
class Article:
    title: str
    summary: str
    url: str = ""
    published: str = ""


@runtime_checkable
class WebSearchBackend(Protocol):
    async def search(self, query: str, max_results: int = 5) -> list[Article]:
        ...


def format_articles(articles: list[Article]) -> str:
    if not articles:
        return ""
    blocks = [
        f"- {a.title} ({a.published}) [{a.url}]\n  {a.summary}"
        for a in articles
    ]
    return "\n".join(blocks)


async def safe_search(backend, query: str, max_results: int = 5) -> str:
    """Search via ``backend`` and return a formatted article block. On ANY failure
    (no backend, rate limit / HTTP 429, network error) return "" so the calling agent
    degrades gracefully to parametric reasoning instead of failing the whole run."""
    if backend is None:
        return ""
    try:
        return format_articles(await backend.search(query, max_results=max_results))
    except Exception as exc:
        log.warning("grounding search failed (%s); falling back to parametric reasoning", exc)
        return ""


class ArxivBackend:
    def __init__(self, client=None, page_size_cap: int = 10):
        self._client = client
        self._page_size_cap = page_size_cap

    async def search(self, query: str, max_results: int = 5) -> list[Article]:
        import arxiv
        if self._client is None:
            page_size = max(1, min(max_results, self._page_size_cap))
            self._client = arxiv.Client(page_size=page_size)

        search = arxiv.Search(query=query, max_results=max_results)
        results = await asyncio.to_thread(list, self._client.results(search))
        return [
            Article(
                title=r.title,
                summary=r.summary,
                url=r.entry_id,
                published=str(r.published),
            )
            for r in results
        ]


class TavilyBackend:
    def __init__(self, api_key: str):
        self._api_key = api_key

    async def search(self, query: str, max_results: int = 5) -> list[Article]:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={"api_key": self._api_key, "query": query, "max_results": max_results},
            )
            resp.raise_for_status()
            data = resp.json()
        return [
            Article(
                title=r.get("title", ""),
                summary=r.get("content", ""),
                url=r.get("url", ""),
            )
            for r in data.get("results", [])
        ]


def build_backend(cfg) -> WebSearchBackend | None:
    b = cfg.grounding.backend
    if b == "arxiv":
        return ArxivBackend()
    if b == "none":
        return None
    if b == "tavily":
        key = os.environ.get("WEB_SEARCH_API_KEY")
        if not key:
            raise RuntimeError("WEB_SEARCH_API_KEY not set — required for Tavily backend.")
        return TavilyBackend(key)
    raise ValueError(
        f"unsupported grounding backend '{b}' (supported: arxiv, none, tavily)"
    )


def is_faithful_grounding(cfg) -> bool:
    return cfg.grounding.backend == "tavily"
