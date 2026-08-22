"""The 2-minute interviewer demo: DuckDB by default, no cloud, no keys.

Loads the bundled synthetic org graph, routes every call through `GraphRetriever` (so no demo
code ever touches a backend directly), and answers "What blocks Project Atlas?" with the full
dependency path, then walks its hierarchy, owners, and risks, and finishes with one
batch-enrichment query standing in for what would otherwise be 4 separate lookups per entity.

`--backend bigquery` runs the identical flow against a BigQuery sandbox dataset populated by
`scripts/setup_bigquery.py` — same `GraphRetriever` calls, same ontology, different backend.

`--ontology-source {file,table}` picks how the ontology registry itself is loaded; `--live-swap`
(DuckDB only) shows the "dynamic" difference: a relationship type INSERTed into the ontology
tables becomes traversable on the very next call, no code change, no restart.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from graph_rag.dialects.duckdb import DuckDbDialect
from graph_rag.ontology import FileOntologySource, Ontology, TableOntologySource
from graph_rag.retriever import GraphRetriever

REPO_ROOT = Path(__file__).parent
ATLAS_ID = "proj-atlas"
ATLAS_CHAIN_DEPTH = 5


def _hr(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def render_path(hit) -> str:
    """Render a BlockerHit's rel_chain/name_chain as a readable farthest-blocker-first path.

    Both arrays grow outward from the anchor as the traversal recurses, so they read
    Atlas-first if used as-is; reversing them and appending the anchor's own name renders the
    path in the natural reading order: farthest blocker -> ... -> the anchor.
    """
    names = list(reversed(hit.name_chain)) + ["Project Atlas"]
    rels = list(reversed(hit.rel_chain))
    parts = [names[0]]
    for rel, name in zip(rels, names[1:], strict=True):
        parts.append(f"-[{rel}]-> {name}")
    return " ".join(parts)


def _duckdb_query_executor(conn):
    """Adapt a DuckDB connection to `TableOntologySource`'s `(sql, params) -> list[dict]` shape."""

    def execute(sql: str, params: dict) -> list[dict]:
        cursor = conn.execute(sql, params)
        columns = [d[0] for d in cursor.description]
        return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]

    return execute


def _seed_duckdb_ontology_tables(conn, ontology: Ontology) -> None:
    """Create and populate `ontology_*` tables in `conn` from an already-loaded Ontology.

    Proves `TableOntologySource` hydrates the identical vocabulary a file source would: the
    seed data below comes straight from the YAML-loaded Ontology object.
    """
    conn.execute("""
        CREATE TABLE ontology_entity_types (
            ontology_name VARCHAR, seq BIGINT, name VARCHAR, description VARCHAR
        )
    """)
    conn.execute("""
        CREATE TABLE ontology_relationship_types (
            ontology_name VARCHAR, seq BIGINT, name VARCHAR, description VARCHAR,
            source_types VARCHAR, target_types VARCHAR, inverse VARCHAR,
            traversal VARCHAR, canonical_direction VARCHAR,
            max_depth BIGINT, fan_out_limit BIGINT
        )
    """)
    conn.execute("""
        CREATE TABLE ontology_semantics (
            ontology_name VARCHAR, seq BIGINT, name VARCHAR, relationship_types VARCHAR
        )
    """)

    for i, et in enumerate(ontology.entity_types):
        conn.execute(
            "INSERT INTO ontology_entity_types VALUES ($ontology_name, $seq, $name, $description)",
            {"ontology_name": ontology.name, "seq": i, "name": et.name, "description": et.description},
        )
    for i, rt in enumerate(ontology.relationship_types):
        conn.execute(
            """
            INSERT INTO ontology_relationship_types VALUES (
                $ontology_name, $seq, $name, $description, $source_types, $target_types,
                $inverse, $traversal, $canonical_direction, $max_depth, $fan_out_limit
            )
            """,
            {
                "ontology_name": ontology.name,
                "seq": i,
                "name": rt.name,
                "description": rt.description,
                "source_types": json.dumps(rt.source_types),
                "target_types": json.dumps(rt.target_types),
                "inverse": rt.inverse,
                "traversal": rt.traversal,
                "canonical_direction": rt.canonical_direction,
                "max_depth": rt.max_depth,
                "fan_out_limit": rt.fan_out_limit,
            },
        )
    for i, sem in enumerate(ontology.semantics):
        conn.execute(
            "INSERT INTO ontology_semantics VALUES ($ontology_name, $seq, $name, $relationship_types)",
            {
                "ontology_name": ontology.name,
                "seq": i,
                "name": sem.name,
                "relationship_types": json.dumps(sem.relationship_types),
            },
        )


def _load_ontology(args, file_ontology: Ontology, conn) -> Ontology:
    if args.ontology_source == "file":
        return file_ontology

    _seed_duckdb_ontology_tables(conn, file_ontology)
    table_source = TableOntologySource(
        execute=_duckdb_query_executor(conn),
        dialect=DuckDbDialect(),
        ontology_name=file_ontology.name,
        version=file_ontology.version,
        table_config=file_ontology.table_config,
    )
    table_ontology = table_source.load()
    assert table_ontology == file_ontology, "table-backed ontology must match the file source"
    print("Loaded ontology from DuckDB tables — identical to the file source (asserted).")
    return table_ontology


def _run_live_swap_demo(retriever: GraphRetriever, conn) -> None:
    """Show the concrete difference between "configurable" and "dynamic".

    INSERTs a new transitive relationship type and a couple of edges of that type, then re-runs
    the identical `find_blockers` call — with no code change and no process restart, it returns
    more rows because the ontology tables, not application code, define what's traversable.
    """
    _hr("Live swap: before")
    before = retriever.find_blockers(ATLAS_ID, max_depth=ATLAS_CHAIN_DEPTH)
    print(f"{len(before)} blockers found for Project Atlas")

    next_seq = conn.execute(
        "SELECT MAX(seq) + 1 FROM ontology_relationship_types WHERE ontology_name = 'org_graph'"
    ).fetchone()[0]
    conn.execute(
        """
        INSERT INTO ontology_relationship_types VALUES (
            'org_graph', $seq, 'escalates', 'The source task escalates risk directly onto the target project.',
            '["Task"]', '["Project"]', NULL, 'transitive', 'source_to_target', 5, NULL
        )
        """,
        {"seq": next_seq},
    )
    conn.execute(
        """
        UPDATE ontology_semantics
        SET relationship_types = '["blocks", "depends_on", "escalates"]'
        WHERE ontology_name = 'org_graph' AND name = 'upstream'
        """
    )
    extra_task_id = conn.execute(
        "SELECT entity_id FROM canonical_entities "
        "WHERE type = 'Task' AND entity_id NOT LIKE 'task-atlas-blocker-%' LIMIT 1"
    ).fetchone()[0]
    conn.execute(
        """
        INSERT INTO entity_relationships
        (source_entity_id, target_entity_id, relationship_type, confidence, created_at)
        VALUES ($source, $target, 'escalates', 1.0, '2025-01-01T00:00:00')
        """,
        {"source": extra_task_id, "target": ATLAS_ID},
    )

    reloaded = TableOntologySource(
        execute=_duckdb_query_executor(conn),
        dialect=DuckDbDialect(),
        ontology_name="org_graph",
        version=retriever.backend.ontology.version,
        table_config=retriever.backend.ontology.table_config,
    ).load()
    retriever.backend.reload_ontology(reloaded)

    _hr("Live swap: after INSERTing a new 'escalates' relationship type — same process, no restart")
    after = retriever.find_blockers(ATLAS_ID, max_depth=ATLAS_CHAIN_DEPTH)
    print(f"{len(after)} blockers found for Project Atlas (was {len(before)})")


def _build_backend(args, ontology: Ontology):
    if args.backend == "duckdb":
        from graph_rag.backends.duckdb_backend import DuckDBGraphBackend

        return DuckDBGraphBackend.from_csv(
            REPO_ROOT / "data" / "entities.csv", REPO_ROOT / "data" / "relationships.csv", ontology
        )

    from graph_rag.backends.bigquery_backend import BigQueryGraphBackend

    return BigQueryGraphBackend(ontology)


def main() -> None:
    parser = argparse.ArgumentParser(description="graph-rag-sql demo")
    parser.add_argument("--backend", choices=["duckdb", "bigquery"], default="duckdb")
    parser.add_argument("--ontology-source", choices=["file", "table"], default="file")
    parser.add_argument(
        "--live-swap", action="store_true",
        help="DuckDB only: demonstrate INSERTing a new relationship type with no restart.",
    )
    args = parser.parse_args()

    start = time.perf_counter()

    file_ontology = Ontology.from_source(
        FileOntologySource(str(REPO_ROOT / "ontology" / "org_graph.yaml"))
    )
    backend = _build_backend(args, file_ontology)

    if args.backend == "duckdb":
        ontology = _load_ontology(args, file_ontology, backend.conn)
        backend.reload_ontology(ontology)
    else:
        ontology = file_ontology

    retriever = GraphRetriever(backend)

    _hr("Graph stats")
    tc = ontology.table_config
    if args.backend == "duckdb":
        n_entities = backend.conn.execute(f"SELECT COUNT(*) FROM {tc.node_table}").fetchone()[0]
        n_edges = backend.conn.execute(f"SELECT COUNT(*) FROM {tc.edge_table}").fetchone()[0]
        print(f"{n_entities} entities, {n_edges} relationships (in-memory DuckDB)")
    else:
        print(f"Querying BigQuery dataset {backend.project_id}.{backend.dataset_id}")

    _hr("What blocks Project Atlas?")
    hits = retriever.find_blockers(ATLAS_ID, max_depth=ATLAS_CHAIN_DEPTH)
    print(f"{len(hits)} blockers found, up to {ATLAS_CHAIN_DEPTH} hops deep:\n")
    for hit in hits:
        print(f"  [distance {hit.distance}] {render_path(hit)}")

    _hr("Project Atlas hierarchy (up to its Goal)")
    hierarchy = retriever.get_hierarchy(ATLAS_ID, direction="up")
    for parent in hierarchy["parents"]:
        print(f"  depth {parent['depth']}: {parent['name']} ({parent['type']})")

    _hr("Owners and risks")
    owners = retriever.get_owners(ATLAS_ID)
    print("Owners:", ", ".join(o.name for o in owners) or "(none)")
    risks = retriever.find_risks(ATLAS_ID)
    print("Risks: ", ", ".join(r.name for r in risks) or "(none)")

    _hr("Batch enrichment: 1 query vs the 12+ queries this replaces")
    if args.backend == "duckdb":
        sample_ids = [ATLAS_ID] + [
            row[0]
            for row in backend.conn.execute(
                f"SELECT {tc.node_id_column} FROM {tc.node_table} "
                f"WHERE type = 'Project' AND {tc.node_id_column} != '{ATLAS_ID}' LIMIT 2"
            ).fetchall()
        ]
    else:
        sample_ids = [ATLAS_ID]
    enriched = retriever.enrich_batch(sample_ids)
    print(f"Enriched {len(sample_ids)} entities with ONE query "
          f"(naive approach: {len(sample_ids)} x 4 lookups = {len(sample_ids) * 4}+ queries)\n")
    for entity_id, result in enriched.items():
        print(
            f"  {entity_id}: {len(result.hierarchy)} ancestors, {len(result.blockers)} blockers, "
            f"{len(result.risks)} risks, {len(result.owners)} owners"
        )

    _hr("Goal status summary (recursive counts)")
    for goal in retriever.goals_summary():
        print(
            f"  {goal['name']} [{goal['status']}]: "
            f"{goal['initiative_count']} initiatives, {goal['project_count']} projects, "
            f"{goal['task_count']} tasks"
        )

    if args.live_swap:
        if args.backend != "duckdb":
            print("\n--live-swap only supported with --backend duckdb; skipping.")
        else:
            _run_live_swap_demo(retriever, backend.conn)

    elapsed = time.perf_counter() - start
    print(f"\nTotal runtime: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
