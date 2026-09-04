"""Traversal-latency (and, on BigQuery, cost) benchmark.

Times `traverse_relationships` (blocks/depends_on, direction "in") at hop depths 1, 2, 3, 5
off the planted "Project Atlas" anchor — whose blocker chain is exactly 5 hops deep, so depth 5
walks the full chain and shorter depths walk a prefix of it — plus one `find_blockers` call and
one `enrich_entities_batch` call over 10 entities. Every row runs through the same
`GraphRetriever` a caller would use.

Methodology, and why each choice is what it is:

* **One untimed warm-up call per row**, before the `--runs` timed iterations, so what the
  timed runs measure is a query the engine has already planned — the steady state a
  long-lived RAG service actually sees, not a cold first request. Measured honestly, this
  changes little on DuckDB at these scales: the first call on a fresh backend runs about
  0.15 ms slower than the steady-state median, which is inside the run-to-run spread. The
  warm-up is insurance against a startup cost leaking into sample #1 (it matters far more on
  a network-backed engine), not a correction that moves the published figures.
* **Median and max, not median and p95.** At the default `--runs 5`, a linear-interpolated p95
  lands between the 4th and 5th sorted samples, so it IS the max to within a hair — printing it
  as "p95" would dress up a 5-sample spread as a tail statistic. `max` says exactly what it is.
  Raise `--runs` to 20+ if you want a figure a percentile label would survive.
* **Every row must return data.** All rows query the planted "Project Atlas" anchor, which
  exists at both scales, so an empty result means the query did not really run. That matters
  most for `enrich_entities_batch`, the one backend method that deliberately swallows
  exceptions and returns `{}` (see `BigQueryGraphBackend.enrich_entities_batch`): without this
  check, a silently-failing enrichment query would benchmark as the FASTEST row in the table,
  and its `total_bytes_processed` would be read stale off the previous call's query job.

Scales: `--scale demo` uses the bundled `data/*.csv` (375 entities / 531 relationships).
`--scale large` generates ~54x more data on the fly into a temp directory — deliberately not
bundled, and never written into the repo.

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
from typing import Any, NamedTuple

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


def _time_runs(fn: Callable[[], Any], runs: int) -> tuple[list[float], Any]:
    """Call `fn` once untimed to warm up, then `runs` more times, timing only those.

    The warm-up absorbs DuckDB query-plan compilation and any first-call lazy import on this
    path, which would otherwise be charged to the first timed sample and inflate the row.

    Returns:
        `(samples_ms, last_result)`. The result comes back so the caller can check the query
        actually returned something — see this module's docstring on silent failures.
    """
    fn()  # warm-up: plan compilation and lazy imports, deliberately not measured
    samples = []
    result = None
    for _ in range(runs):
        start = perf_counter()
        result = fn()
        samples.append((perf_counter() - start) * 1000)
    return samples, result


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


class BenchRow(NamedTuple):
    """One benchmarked operation.

    `returned_data` is False when the call came back empty off the planted Atlas anchor, which
    can only mean the query didn't really run. Such a row is reported as FAILED rather than as
    a fast one, and its `total_bytes` is dropped — on BigQuery it would otherwise be read stale
    off the preceding call's query job.
    """

    label: str
    samples_ms: list[float]
    total_bytes: int | None
    returned_data: bool


def _bytes_processed(retriever: GraphRetriever) -> int | None:
    """Bytes BigQuery billed for the most recent call, or None off a backend that doesn't track it."""
    job = getattr(retriever.backend, "last_query_job", None)
    return getattr(job, "total_bytes_processed", None)


def _bigquery_cost_usd(total_bytes: int) -> float:
    billed_bytes = max(total_bytes, BIGQUERY_MIN_BILLED_BYTES_PER_TABLE)
    tebibytes = billed_bytes / (1024**4)
    return tebibytes * BIGQUERY_PRICE_PER_TIB_USD


def _row(label: str, fn: Callable[[], Any], runs: int, retriever: GraphRetriever) -> BenchRow:
    """Time `fn` and record whether it returned anything at all."""
    samples, result = _time_runs(fn, runs)
    returned_data = bool(result)
    return BenchRow(
        label=label,
        samples_ms=samples,
        # Stale bytes off the previous call would be worse than no number at all.
        total_bytes=_bytes_processed(retriever) if returned_data else None,
        returned_data=returned_data,
    )


def _benchmark_rows(retriever: GraphRetriever, runs: int) -> list[BenchRow]:
    rows: list[BenchRow] = []

    for depth in HOP_DEPTHS:
        filters = GraphFilters(
            entity_id=ATLAS_ID, rel_type=UPSTREAM_REL_TYPES, rel_direction="in", rel_max_depth=depth
        )
        rows.append(
            _row(
                f"{depth}-hop traversal",
                lambda filters=filters: retriever.traverse(filters),
                runs,
                retriever,
            )
        )

    rows.append(
        _row(
            "find_blockers (5-hop)",
            lambda: retriever.find_blockers(ATLAS_ID, max_depth=5),
            runs,
            retriever,
        )
    )

    # proj-0..proj-8 are generated at both "demo" and "large" scale — see generator.py.
    sample_ids = [ATLAS_ID] + [f"proj-{i}" for i in range(ENRICH_BATCH_SIZE - 1)]
    rows.append(
        _row(
            f"enrich_entities_batch({ENRICH_BATCH_SIZE} ids)",
            lambda: retriever.enrich_batch(sample_ids),
            runs,
            retriever,
        )
    )

    return rows


def _print_table(backend_name: str, rows: list[BenchRow], runs: int) -> None:
    """Print the results as a markdown table.

    The `max (ms, n=N)` header names the statistic actually computed — `max(samples)`. See the
    module docstring for why this is not labelled p95 at small `--runs`.
    """
    is_bigquery = backend_name == "bigquery"
    header = f"| Traversal | Median (ms) | Max (ms, n={runs}) |"
    sep = "|---|---|---|"
    if is_bigquery:
        header += " Bytes billed | Est. cost |"
        sep += "---|---|"
    print(header)
    print(sep)

    for row in rows:
        if not row.returned_data:
            line = f"| {row.label} | FAILED | FAILED |"
            if is_bigquery:
                line += " n/a | n/a |"
            print(line)
            continue
        line = (
            f"| {row.label} | {statistics.median(row.samples_ms):.2f} "
            f"| {max(row.samples_ms):.2f} |"
        )
        if is_bigquery:
            if row.total_bytes is None:
                line += " n/a | n/a |"
            else:
                line += f" {row.total_bytes:,} B | ${_bigquery_cost_usd(row.total_bytes):.6f} |"
        print(line)

    failed = [row.label for row in rows if not row.returned_data]
    if failed:
        print()
        print(
            "WARNING: these rows returned no data and are NOT valid measurements: "
            + ", ".join(failed)
            + "."
        )
        print(
            "Every row queries the planted 'Project Atlas' anchor and must return "
            "something, so an empty result means the query did not run. "
            "`BigQueryGraphBackend.enrich_entities_batch` deliberately catches exceptions "
            "and returns an empty dict — a failure there would otherwise time as the "
            "FASTEST row in this table. Do not publish these numbers; fix the query first."
        )

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
    _print_table(args.backend, rows, args.runs)

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return
    _save_latency_chart(rows)


def _save_latency_chart(rows: list[BenchRow]) -> None:
    """Save `benchmark_latency.png` — only runs if the optional `[bench]` extra is installed.

    Rows that returned no data are omitted rather than charted as fast ones.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plotted = [row for row in rows if row.returned_data]
    labels = [row.label for row in plotted]
    medians = [statistics.median(row.samples_ms) for row in plotted]

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
