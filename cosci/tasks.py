"""Async priority queue over Task. Lower priority value = scheduled sooner; FIFO within a level."""
from __future__ import annotations
import asyncio
import itertools
from cosci.models import Task

class GlobalTaskQueue:
    def __init__(self) -> None:
        self._q: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._counter = itertools.count()

    async def put(self, task: Task) -> None:
        # (priority, insertion_order) makes ordering total without comparing Task objects
        await self._q.put((task.priority, next(self._counter), task))

    async def get(self) -> Task:
        _, _, task = await self._q.get()
        return task

    def empty(self) -> bool:
        return self._q.empty()

    def qsize(self) -> int:
        return self._q.qsize()
