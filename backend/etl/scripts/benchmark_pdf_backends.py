"""
Benchmark harness for comparing PDF extraction backends.

Measures extraction time, memory usage, and output quality across
Marker and Docling backends.

Usage:
    uv run python -m backend.etl.scripts.benchmark_pdf_backends \
        --pdf-dir backend/tests/etl/fixtures/pdfs \
        --sample-size 2 \
        --backends marker docling \
        --output reports/benchmark.json
"""

import argparse
import json
import statistics
import time
import tracemalloc
from pathlib import Path

import psutil
import tiktoken
from loguru import logger

from backend.etl.utils.pdf_backends import get_backend


def benchmark_single_pdf(
    backend_name: str,
    pdf_path: Path,
    backend_config: dict | None = None,
) -> dict:
    """Benchmark a single backend on a single PDF."""
    backend = get_backend(backend_name, config=backend_config)
    process = psutil.Process()

    # Measure memory and time
    rss_before = process.memory_info().rss
    tracemalloc.start()
    start = time.perf_counter()

    result = backend.extract(pdf_path)

    elapsed = time.perf_counter() - start
    _, peak_traced = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = process.memory_info().rss

    # Count output tokens
    output_chars = len(result.text) if result.success else 0
    output_tokens = 0
    if result.success and result.text:
        enc = tiktoken.get_encoding("cl100k_base")
        output_tokens = len(enc.encode(result.text))

    backend.cleanup()

    return {
        "pdf": str(pdf_path),
        "backend": backend_name,
        "success": result.success,
        "error": result.error,
        "elapsed_seconds": round(elapsed, 3),
        "peak_traced_memory_mb": round(peak_traced / (1024 * 1024), 2),
        "rss_delta_mb": round((rss_after - rss_before) / (1024 * 1024), 2),
        "output_chars": output_chars,
        "output_tokens": output_tokens,
        "pages": result.pages,
    }


def aggregate_results(per_pdf_results: list[dict]) -> dict:
    """Compute aggregate statistics for a single backend."""
    successes = [r for r in per_pdf_results if r["success"]]
    failures = [r for r in per_pdf_results if not r["success"]]

    times = [r["elapsed_seconds"] for r in successes]
    memories = [r["peak_traced_memory_mb"] for r in successes]
    chars = [r["output_chars"] for r in successes]
    tokens = [r["output_tokens"] for r in successes]

    def safe_stats(values: list[float]) -> dict:
        if not values:
            return {"mean": 0, "median": 0, "p95": 0, "max": 0}
        sorted_vals = sorted(values)
        p95_idx = min(int(len(sorted_vals) * 0.95), len(sorted_vals) - 1)
        return {
            "mean": round(statistics.mean(values), 3),
            "median": round(statistics.median(values), 3),
            "p95": round(sorted_vals[p95_idx], 3),
            "max": round(max(values), 3),
        }

    return {
        "total_pdfs": len(per_pdf_results),
        "success_count": len(successes),
        "failure_count": len(failures),
        "success_rate": round(len(successes) / len(per_pdf_results), 3)
        if per_pdf_results
        else 0,
        "time_stats": safe_stats(times),
        "memory_stats": safe_stats(memories),
        "output_chars_stats": safe_stats(chars),
        "output_tokens_stats": safe_stats(tokens),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PDF extraction backends")
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        required=True,
        help="Directory containing PDF files to benchmark",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Number of PDFs to sample (0 = all)",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["marker", "docling"],
        help="Backends to benchmark",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/benchmark.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    # Find PDFs
    pdf_files = sorted(args.pdf_dir.glob("**/*.pdf"))
    if not pdf_files:
        logger.error(f"No PDFs found in {args.pdf_dir}")
        return

    if args.sample_size > 0:
        pdf_files = pdf_files[: args.sample_size]

    logger.info(f"Benchmarking {len(pdf_files)} PDFs with backends: {args.backends}")

    per_pdf_results: list[dict] = []
    summaries: dict[str, dict] = {}

    for backend_name in args.backends:
        logger.info(f"--- Benchmarking: {backend_name} ---")
        backend_results: list[dict] = []

        for pdf_path in pdf_files:
            logger.info(f"  Processing: {pdf_path.name}")
            try:
                result = benchmark_single_pdf(backend_name, pdf_path)
                backend_results.append(result)
                per_pdf_results.append(result)
                status = "OK" if result["success"] else "FAIL"
                logger.info(
                    f"  {status}: {result['elapsed_seconds']}s, "
                    f"{result['output_tokens']} tokens"
                )
            except Exception as e:
                logger.error(f"  Error benchmarking {pdf_path.name}: {e}")
                backend_results.append(
                    {
                        "pdf": str(pdf_path),
                        "backend": backend_name,
                        "success": False,
                        "error": str(e),
                        "elapsed_seconds": 0,
                        "peak_traced_memory_mb": 0,
                        "rss_delta_mb": 0,
                        "output_chars": 0,
                        "output_tokens": 0,
                        "pages": 0,
                    }
                )

        summaries[backend_name] = aggregate_results(backend_results)

    # Write report
    report = {"per_pdf": per_pdf_results, "summaries": summaries}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Benchmark report written to {args.output}")

    # Print summary table
    logger.info("\n=== Summary ===")
    for name, summary in summaries.items():
        logger.info(
            f"{name}: "
            f"{summary['success_count']}/{summary['total_pdfs']} success, "
            f"median time={summary['time_stats']['median']}s, "
            f"mean tokens={summary['output_tokens_stats']['mean']}"
        )


if __name__ == "__main__":
    main()
