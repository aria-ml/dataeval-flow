"""Launch the local execution service."""

import argparse
from pathlib import Path

import uvicorn

from dataeval_flow_service._app import create_app


def main() -> None:
    """Serve one local CPU worker with explicit persistent storage roots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    uvicorn.run(create_app(args.data, args.output, args.cache), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
