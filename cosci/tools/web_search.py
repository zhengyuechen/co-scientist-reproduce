"""Pluggable literature-search layer for the Co-Scientist system.

NOTE: This backend layer is OURS — the paper used broad web search (Tavily / Google).
      arXiv-only is a deliberate fidelity compromise: free, no key required, but
      limited to preprints. Use a Tavily/Serper/Brave backend for faithful grounding.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


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


class ArxivBackend:
    def __init__(self, client=None):
        self._client = client

    async def search(self, query: str, max_results: int = 5) -> list[Article]:
        if self._client is None:
            import arxiv
            self._client = arxiv.Client()

        import arxiv  # noqa: F811 — needed for arxiv.Search
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
