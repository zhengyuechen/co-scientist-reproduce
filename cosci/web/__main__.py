"""Launch the web UI:  python -m cosci.web  (then open http://127.0.0.1:8000)."""
from __future__ import annotations

import argparse

import uvicorn


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="cosci.web", description="Co-Scientist web UI.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000).")
    args = parser.parse_args(argv)
    uvicorn.run("cosci.web.app:app", host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
