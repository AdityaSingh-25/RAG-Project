"""Concurrent load test for /query and /query/stream.

Fires ``--concurrency`` workers, each running ``--requests`` queries against
the local API, and reports latency percentiles + throughput. The point isn't
benchmarking against a perfect baseline — it's to actually measure what
happens when N requests arrive at once, so we can pick a sensible
concurrency cap and notice regressions over time.

Examples:

    # 32 concurrent clients, 5 queries each, against a running API.
    python scripts/load_test.py --concurrency 32 --requests 5

    # Exercise the SSE path instead — measures total stream duration end-to-end.
    python scripts/load_test.py --mode stream --concurrency 16 --requests 3

    # Pick your own questions (one per line); useful for hitting the cache.
    python scripts/load_test.py --questions-file my_questions.txt

The script does not start the API — bring it up separately with
``uvicorn rag_engine.api.main:app`` (or ``rag-api``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import httpx

DEFAULT_QUESTIONS = [
    "What are the main themes in the ingested documents?",
    "Which sources support the answer most strongly?",
    "Summarise the architecture from the corpus.",
    "What should the system refuse to answer?",
    "How does retrieval combine BM25 and dense embeddings?",
]


@dataclass
class Result:
    ok: bool
    duration_ms: float
    status_code: int
    error: str | None = None


async def _run_query(
    client: httpx.AsyncClient, url: str, question: str, bypass_cache: bool
) -> Result:
    started = time.perf_counter()
    try:
        resp = await client.post(
            url,
            json={"question": question, "bypass_cache": bypass_cache},
            timeout=120.0,
        )
        duration_ms = (time.perf_counter() - started) * 1000
        if resp.status_code != 200:
            return Result(ok=False, duration_ms=duration_ms, status_code=resp.status_code, error=resp.text[:200])
        return Result(ok=True, duration_ms=duration_ms, status_code=200)
    except Exception as exc:
        duration_ms = (time.perf_counter() - started) * 1000
        return Result(ok=False, duration_ms=duration_ms, status_code=0, error=str(exc))


async def _run_stream(
    client: httpx.AsyncClient, url: str, question: str, bypass_cache: bool
) -> Result:
    started = time.perf_counter()
    try:
        async with client.stream(
            "POST",
            url,
            json={"question": question, "bypass_cache": bypass_cache},
            timeout=120.0,
        ) as resp:
            saw_done = False
            async for line in resp.aiter_lines():
                if line.startswith("event: done"):
                    saw_done = True
            duration_ms = (time.perf_counter() - started) * 1000
            if resp.status_code != 200:
                return Result(ok=False, duration_ms=duration_ms, status_code=resp.status_code, error="non-200 stream")
            if not saw_done:
                return Result(ok=False, duration_ms=duration_ms, status_code=resp.status_code, error="no done event")
            return Result(ok=True, duration_ms=duration_ms, status_code=200)
    except Exception as exc:
        duration_ms = (time.perf_counter() - started) * 1000
        return Result(ok=False, duration_ms=duration_ms, status_code=0, error=str(exc))


async def _worker(
    client: httpx.AsyncClient,
    url: str,
    questions: list[str],
    n_requests: int,
    bypass_cache: bool,
    mode: str,
    worker_id: int,
) -> list[Result]:
    out: list[Result] = []
    for i in range(n_requests):
        q = questions[(worker_id + i) % len(questions)]
        if mode == "stream":
            out.append(await _run_stream(client, url, q, bypass_cache))
        else:
            out.append(await _run_query(client, url, q, bypass_cache))
    return out


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _load_questions(path: Path | None) -> list[str]:
    if path is None:
        return DEFAULT_QUESTIONS
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if not lines:
        raise SystemExit(f"--questions-file {path} produced no questions")
    return lines


def _print_report(mode: str, concurrency: int, requests: int, results: Iterable[Result], wall_seconds: float) -> int:
    results = list(results)
    n = len(results)
    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    durations = [r.duration_ms for r in ok]

    if durations:
        p50 = _percentile(durations, 50)
        p95 = _percentile(durations, 95)
        p99 = _percentile(durations, 99)
        mean = statistics.mean(durations)
        mx = max(durations)
    else:
        p50 = p95 = p99 = mean = mx = 0.0

    throughput = n / wall_seconds if wall_seconds > 0 else 0.0
    success_rate = (len(ok) / n * 100) if n else 0.0

    print()
    print(f"=== load test · mode={mode} · concurrency={concurrency} · requests/worker={requests} ===")
    print(f"  total              {n}")
    print(f"  ok                 {len(ok)}  ({success_rate:.1f}%)")
    print(f"  failed             {len(failed)}")
    print(f"  wall time          {wall_seconds:.2f}s")
    print(f"  throughput         {throughput:.2f} req/s")
    print()
    print("  latency (ms, successful requests):")
    print(f"    mean             {mean:.1f}")
    print(f"    p50              {p50:.1f}")
    print(f"    p95              {p95:.1f}")
    print(f"    p99              {p99:.1f}")
    print(f"    max              {mx:.1f}")

    if failed:
        print()
        print("  first failures:")
        for r in failed[:5]:
            print(f"    [status={r.status_code}] {r.error!r}")

    # Non-zero exit if anything failed — useful when wired into CI later.
    return 0 if not failed else 1


async def _amain(args: argparse.Namespace) -> int:
    questions = _load_questions(Path(args.questions_file) if args.questions_file else None)
    url = f"{args.url.rstrip('/')}{'/query/stream' if args.mode == 'stream' else '/query'}"

    limits = httpx.Limits(max_connections=args.concurrency * 2, max_keepalive_connections=args.concurrency)
    async with httpx.AsyncClient(limits=limits) as client:
        started = time.perf_counter()
        tasks = [
            _worker(
                client,
                url,
                questions,
                args.requests,
                args.bypass_cache,
                args.mode,
                worker_id=w,
            )
            for w in range(args.concurrency)
        ]
        batches = await asyncio.gather(*tasks)
        wall = time.perf_counter() - started

    results = [r for batch in batches for r in batch]
    if args.json:
        json.dump(
            [r.__dict__ for r in results],
            sys.stdout,
            indent=2,
            default=str,
        )
        print()
    return _print_report(args.mode, args.concurrency, args.requests, results, wall)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--mode", choices=["query", "stream"], default="query", help="Hit /query or /query/stream")
    parser.add_argument("--concurrency", type=int, default=8, help="Number of concurrent workers")
    parser.add_argument("--requests", type=int, default=5, help="Requests per worker")
    parser.add_argument("--bypass-cache", action="store_true", help="Set bypass_cache=true on every request")
    parser.add_argument("--questions-file", help="One question per line")
    parser.add_argument("--json", action="store_true", help="Also dump raw results as JSON before the summary")
    args = parser.parse_args()
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    sys.exit(main())
