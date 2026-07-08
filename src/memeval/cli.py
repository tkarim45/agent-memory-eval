"""Command line for the agent-memory benchmark.

    agent-memory-eval                    # real Claude on Bedrock + MiniLM retriever
    agent-memory-eval --k 4
    agent-memory-eval --backend anthropic
    agent-memory-eval --offline          # fake agent + hashing retriever, no keys
    agent-memory-eval --json results.json

A real run makes one summary call + one answer call per (strategy × question).
"""
from __future__ import annotations

import argparse
import json
import sys

from .benchmark import format_report, run


def _build(backend: str, offline: bool):
    if offline:
        from .benchmark import FakeCompleter
        from .embed import HashingEmbedder
        return FakeCompleter(), HashingEmbedder()
    from .embed import SentenceTransformerEmbedder
    client = __import__("memeval.llm", fromlist=["build_client"]).build_client(backend)
    return client, SentenceTransformerEmbedder()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Benchmark agent memory strategies on a recall test.")
    ap.add_argument("--k", type=int, default=4, help="retrieval depth for the vector-memory strategy")
    ap.add_argument("--backend", default="bedrock", choices=["bedrock", "anthropic"])
    ap.add_argument("--offline", action="store_true", help="fake agent + hashing retriever, no keys")
    ap.add_argument("--json", metavar="PATH", help="write full results as JSON")
    args = ap.parse_args(argv)

    client, embedder = _build(args.backend, args.offline)
    result = run(client, embedder, k=args.k)

    print(format_report(result))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
