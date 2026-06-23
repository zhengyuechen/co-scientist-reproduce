import asyncio
import json

import pytest

from cosci import run_log


def test_emit_is_noop_when_unbound():
    # No logger bound in this context -> emit must not raise and writes nothing.
    run_log.emit("task", tick=1, agent="generation")


def test_bind_writes_events_and_read_is_incremental(tmp_path):
    path = run_log.events_path(str(tmp_path), "2026-06-22_000000_goal")

    async def run():
        run_log.bind(path)
        run_log.emit("run_started", goal="g")
        run_log.emit("task", tick=1, agent="generation", action="create_initial_hypotheses")
        run_log.emit("run_done", hypotheses=6, matches=8)

    asyncio.run(run())

    all_events = run_log.read_events(path)
    assert [e["event"] for e in all_events] == ["run_started", "task", "run_done"]
    assert all_events[1]["agent"] == "generation" and all_events[1]["tick"] == 1
    assert all("time" in e for e in all_events)
    # incremental read returns only events after the cursor
    assert [e["event"] for e in run_log.read_events(path, since=2)] == ["run_done"]
    assert run_log.read_events(path, since=3) == []


def test_read_events_missing_file_returns_empty(tmp_path):
    assert run_log.read_events(tmp_path / "nope.jsonl") == []


def test_concurrent_runs_get_isolated_logs(tmp_path):
    pa = run_log.events_path(str(tmp_path), "runA")
    pb = run_log.events_path(str(tmp_path), "runB")

    async def one(path, tag):
        run_log.bind(path)               # binds in THIS task's context only
        run_log.emit("run_started", goal=tag)
        await asyncio.sleep(0)
        run_log.emit("run_done", goal=tag)

    async def both():
        await asyncio.gather(
            asyncio.create_task(one(pa, "A")),
            asyncio.create_task(one(pb, "B")),
        )

    asyncio.run(both())
    a = run_log.read_events(pa)
    b = run_log.read_events(pb)
    assert {e["goal"] for e in a} == {"A"} and len(a) == 2
    assert {e["goal"] for e in b} == {"B"} and len(b) == 2
