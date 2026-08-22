"""Traversal-latency (and, on BigQuery, cost) benchmark.

Times `traverse_relationships` (blocks/depends_on, direction "in") at hop depths 1, 2, 3, 5
off the planted "Project Atlas" anchor — whose blocker chain is exactly 5 hops deep, so depth 5
walks the full chain and shorter depths walk a prefix of it — plus one `find_blockers` call and
one `enrich_entities_batch` call over 10 entities. Every row is run `--runs` times through the
same `GraphRetriever` a caller would use, and reports median + p95 wall-clock milliseconds.

On `--backend bigquery`, each row additionally reports the `total_bytes_processed` BigQuery
billed for the LAST of those runs (read off `BigQueryGraphBackend.last_query_job`) and an
estimated on-demand cost. `use_query_cache=False` is passed to the backend so a cache hit on a
repeated run can't report `total_bytes_processed = 0` and understate the real cost of a cold
traversal.

Usage:
    python scripts/benchmark.py --backend duckdb --scale demo
    python scripts/benchmark.py --backend bigquery --scale demo --runs 10
"""

from __future__ import annotations

import argparse
import statistics
import tempfile
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

from graph_rag.generator import generate_org_graph, write_csvs
from graph_rag.models import GraphFilters
from graph_rag.ontology import FileOntologySource, Ontology
from graph_rag.retriever import GraphRetriever

REPO_ROOT = Path(__file__).parent.parent
ATLAS_ID = "proj-atlas"
UPSTREAM_REL_TYPES = ["blocks", "depends_on"]
HOP_DEPTHS = [1, 2, 3, 5]
ENRICH_BATCH_SIZE = 10

# BigQuery on-demand list price as of mid-2026 — see
# https://cloud.google.com/bigquery/pricing#on_demand_pricing
BIGQUERY_PRICE_PER_TIB_USD = 6.25
# BigQuery bills at least 10 MB per table scanned per query, even on a tiny hit.
BIGQUERY_MIN_BILLED_BYTES_PER_TABLE = 10 * 1024 * 1024


def _percentile(samples: list[float], pct: float) -> float:
    """Linear-interpolated percentile (matches `numpy.percentile`'s default)."""
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100)
    lo, hi = int(rank), min(int(rank) + 1, len(ordered) - 1)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def _time_runs(fn: Callable[[], Any], runs: int) -> list[float]:
    samples = []
    for _ in range(runs):
        start = perf_counter()
        fn()
        samples.append((perf_counter() - start) * 1000)
    return samples


def _load_ontology() -> Ontology:
    return Ontology.from_source(FileOntologySource(str(REPO_ROOT / "ontology" / "org_graph.yaml")))


def _build_duckdb_retriever(scale: str) -> GraphRetriever:
    from graph_rag.backends.duckdb_backend import DuckDBGraphBackend

    ontology = _load_ontology()
    if scale == "demo":
        entities_csv = REPO_ROOT / "data" / "entities.csv"
        relationships_csv = REPO_ROOT / "data" / "relationships.csv"
    else:
        entities, relationships = generate_org_graph(scale="large")
        tmp_dir = Path(tempfile.mkdtemp(prefix="graph_rag_bench_"))
        write_csvs(entities, relationships, tmp_dir)
        entities_csv, relationships_csv = tmp_dir / "entities.csv", tmp_dir / "relationships.csv"

    backend = DuckDBGraphBackend.from_csv(entities_csv, relationships_csv, ontology)
    return GraphRetriever(backend)


def _build_bigquery_retriever(scale: str) -> GraphRetriever:
    from graph_rag.backends.bigquery_backend import BigQueryGraphBackend

    if scale == "large":
        print(
            "note: --scale large does not regenerate BigQuery data; "
            "run scripts/setup_bigquery.py against a dataset already loaded at that scale.\n"
        )
    backend = BigQueryGraphBackend(_load_ontology(), use_query_cache=False)
    return GraphRetriever(backend)


def _bytes_processed(retriever: GraphRetriever) -> int | None:
    """Bytes BigQuery billed for the most recent call, or None off a backend that doesn't track it."""
    job = getattr(retriever.backend, "last_query_job", None)
    return getattr(job, "total_bytes_processed", None)


def _bigquery_cost_usd(total_bytes: int) -> float:
    billed_bytes = max(total_bytes, BIGQUERY_MIN_BILLED_BYTES_PER_TABLE)
    tebibytes = billed_bytes / (1024**4)
    return tebibytes * BIGQUERY_PRICE_PER_TIB_USD


def _benchmark_rows(retriever: GraphRetriever, runs: int) -> list[tuple[str, list[float], int | None]]:
    rows: list[tuple[str, list[float], int | None]] = []

    for depth in HOP_DEPTHS:
        filters = GraphFilters(
            entity_id=ATLAS_ID, rel_type=UPSTREAM_REL_TYPES, rel_direction="in", rel_max_depth=depth
        )
        samples = _time_runs(lambda filters=filters: retriever.traverse(filters), runs)
        rows.append((f"{depth}-hop traversal", samples, _bytes_processed(retriever)))

    samples = _time_runs(lambda: retriever.find_blockers(ATLAS_ID, max_depth=5), runs)
    rows.append(("find_blockers (5-hop)", samples, _bytes_processed(retriever)))

    # proj-0..proj-8 are generated at both "demo" and "large" scale — see generator.py.
    sample_ids = [ATLAS_ID] + [f"proj-{i}" for i in range(ENRICH_BATCH_SIZE - 1)]
    samples = _time_runs(lambda: retriever.enrich_batch(sample_ids), runs)
    rows.append((f"enrich_entities_batch({ENRICH_BATCH_SIZE} ids)", samples, _bytes_processed(retriever)))

    return rows


def _print_table(backend_name: str, rows: list[tuple[str, list[float], int | None]]) -> None:
    is_bigquery = backend_name == "bigquery"
    header = "| Traversal | Median (ms) | p95 (ms) |"
    sep = "|---|---|---|"
    if is_bigquery:
        header += " Bytes billed | Est. cost |"
        sep += "---|---|"
    print(header)
    print(sep)

    for label, samples, total_bytes in rows:
        line = f"| {label} | {statistics.median(samples):.2f} | {_percentile(samples, 95):.2f} |"
        if is_bigquery:
            if total_bytes is None:
                line += " n/a | n/a |"
            else:
                line += f" {total_bytes:,} B | ${_bigquery_cost_usd(total_bytes):.6f} |"
        print(line)

    if is_bigquery:
        print(
            f"\nCost estimated at the on-demand list price of ${BIGQUERY_PRICE_PER_TIB_USD}/TiB "
            "(https://cloud.google.com/bigquery/pricing#on_demand_pricing), applying BigQuery's "
            f"{BIGQUERY_MIN_BILLED_BYTES_PER_TABLE // (1024 * 1024)} MB per-table minimum "
            "billing per query. `use_query_cache=False` on the backend so a cache hit can't "
            "report 0 bytes and understate a cold traversal's real cost."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark graph-rag-sql traversal latency and cost.")
    parser.add_argument("--backend", choices=["duckdb", "bigquery"], default="duckdb")
    parser.add_argument("--scale", choices=["demo", "large"], default="demo")
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    retriever = (
        _build_duckdb_retriever(args.scale)
        if args.backend == "duckdb"
        else _build_bigquery_retriever(args.scale)
    )
    rows = _benchmark_rows(retriever, args.runs)
    _print_table(args.backend, rows)

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return
    _save_latency_chart(rows)


def _save_latency_chart(rows: list[tuple[str, list[float], int | None]]) -> None:
    """Save `benchmark_latency.png` — only runs if the optional `[bench]` extra is installed."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [label for label, _, _ in rows]
    medians = [statistics.median(samples) for _, samples, _ in rows]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(labels, medians, color="#4C72B0")
    ax.set_xlabel("Median latency (ms)")
    ax.set_title("graph-rag-sql traversal latency")
    fig.tight_layout()
    out_path = REPO_ROOT / "benchmark_latency.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
