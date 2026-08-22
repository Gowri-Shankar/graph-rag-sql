"""The 2-minute interviewer demo: pure DuckDB, no cloud, no keys.

Loads the bundled synthetic org graph into in-memory DuckDB via `DuckDBGraphBackend` and
answers "What blocks Project Atlas?" with the full dependency path, then walks its hierarchy,
owners, and risks, and finishes with one batch-enrichment query standing in for what would
otherwise be 4 separate lookups per entity.

A later milestone adds a `GraphRetriever` facade and a BigQuery backend; this milestone calls
`DuckDBGraphBackend` directly.
"""

from __future__ import annotations

import time
from pathlib import Path

from graph_rag.backends.duckdb_backend import DuckDBGraphBackend
from graph_rag.ontology import FileOntologySource, Ontology

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


def main() -> None:
    start = time.perf_counter()

    ontology = Ontology.from_source(FileOntologySource(str(REPO_ROOT / "ontology" / "org_graph.yaml")))
    backend = DuckDBGraphBackend.from_csv(
        REPO_ROOT / "data" / "entities.csv", REPO_ROOT / "data" / "relationships.csv", ontology
    )

    _hr("Graph stats")
    n_entities = backend.conn.execute(f"SELECT COUNT(*) FROM {ontology.table_config.node_table}").fetchone()[0]
    n_edges = backend.conn.execute(f"SELECT COUNT(*) FROM {ontology.table_config.edge_table}").fetchone()[0]
    print(f"{n_entities} entities, {n_edges} relationships (in-memory DuckDB)")

    _hr("What blocks Project Atlas?")
    hits = backend.find_blockers(ATLAS_ID, max_depth=ATLAS_CHAIN_DEPTH)
    print(f"{len(hits)} blockers found, up to {ATLAS_CHAIN_DEPTH} hops deep:\n")
    for hit in hits:
        print(f"  [distance {hit.distance}] {render_path(hit)}")

    _hr("Project Atlas hierarchy (up to its Goal)")
    hierarchy = backend.get_entity_hierarchy(ATLAS_ID, direction="up")
    for parent in hierarchy["parents"]:
        print(f"  depth {parent['depth']}: {parent['name']} ({parent['type']})")

    _hr("Owners and risks")
    owners = backend.get_entity_owners(ATLAS_ID)
    print("Owners:", ", ".join(o.name for o in owners) or "(none)")
    risks = backend.find_risks_for_entity(ATLAS_ID)
    print("Risks: ", ", ".join(r.name for r in risks) or "(none)")

    _hr("Batch enrichment: 1 query vs the 12+ queries this replaces")
    sample_ids = [ATLAS_ID] + [
        row[0]
        for row in backend.conn.execute(
            f"SELECT {ontology.table_config.node_id_column} "
            f"FROM {ontology.table_config.node_table} WHERE type = 'Project' "
            f"AND {ontology.table_config.node_id_column} != '{ATLAS_ID}' LIMIT 2"
        ).fetchall()
    ]
    enriched = backend.enrich_entities_batch(sample_ids)
    print(f"Enriched {len(sample_ids)} entities with ONE query "
          f"(naive approach: {len(sample_ids)} x 4 lookups = {len(sample_ids) * 4}+ queries)\n")
    for entity_id, result in enriched.items():
        print(
            f"  {entity_id}: {len(result.hierarchy)} ancestors, {len(result.blockers)} blockers, "
            f"{len(result.risks)} risks, {len(result.owners)} owners"
        )

    _hr("Goal status summary (recursive counts)")
    for goal in backend.get_goals_status_summary():
        print(
            f"  {goal['name']} [{goal['status']}]: "
            f"{goal['initiative_count']} initiatives, {goal['project_count']} projects, "
            f"{goal['task_count']} tasks"
        )

    elapsed = time.perf_counter() - start
    print(f"\nTotal runtime: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
