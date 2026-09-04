"""THE PORTABILITY PROOF — in both of its dimensions.

VOCABULARY portability: all three patterns run against a second ontology (Service/Incident/
Team) that shares no type names with the org graph.

SCHEMA portability: all eight `GraphBackend` methods run against a third ontology with the
same vocabulary but a different physical schema — different table names, different column
names, and only the four REQUIRED node columns, which is the shape README's "expose two
views" example teaches. For a long time both fixtures copied org_graph's `table_config`
verbatim, so this suite proved type-name portability and nothing whatsoever about schema
portability; seven of the eight methods in fact failed on a four-column node table.

Both must pass with ZERO changes to any file under `src/graph_rag/patterns/` beyond keeping
them schema-neutral — if they don't, the patterns still carry coupling; the fix belongs in the
patterns or the backends, never in this test.
"""

from __future__ import annotations

from datetime import datetime

from graph_rag.backends.duckdb_backend import DuckDBGraphBackend
from graph_rag.generator import write_csvs
from graph_rag.models import Entity, Relationship
from graph_rag.ontology import FileOntologySource, Ontology

TINY_DOMAIN_PATH = "tests/fixtures/tiny_domain.yaml"
ALT_SCHEMA_PATH = "tests/fixtures/tiny_domain_alt_schema.yaml"


def _entity(entity_id: str, name: str, type_: str, status: str) -> Entity:
    return Entity(
        entity_id=entity_id, name=name, type=type_, status=status,
        created_at=datetime(2025, 1, 1),
    )


def _edge(source: str, target: str, rel_type: str) -> Relationship:
    return Relationship(
        source_entity_id=source,
        target_entity_id=target,
        relationship_type=rel_type,
        created_at=datetime(2025, 1, 1),
    )


# One graph, loaded two ways: through the generator's wide CSV schema below, and through the
# README's minimal four-column views in `_build_alt_schema_backend`. Shared so the two are
# comparable row for row.
_NODES = [
    ("svc-a", "Checkout Service", "Service", "degraded"),
    ("svc-b", "Payments Service", "Service", "healthy"),
    ("svc-c", "Ledger Service", "Service", "healthy"),
    ("team-1", "Platform Team", "Team", "active"),
    ("inc-1", "Root Incident", "Incident", "open"),
    ("inc-2", "Middle Incident", "Incident", "open"),
    ("inc-3", "Leaf Incident", "Incident", "resolved"),
]
_EDGES = [
    ("svc-a", "svc-b", "runs_on"),
    ("svc-b", "svc-c", "runs_on"),
    ("inc-3", "inc-2", "caused_by"),
    ("inc-2", "inc-1", "caused_by"),
    ("team-1", "svc-a", "on_call_for"),
]


def _build_tiny_domain_backend(tmp_path) -> DuckDBGraphBackend:
    ontology = Ontology.from_source(FileOntologySource(TINY_DOMAIN_PATH))
    entities = [_entity(*node) for node in _NODES]
    relationships = [_edge(*edge) for edge in _EDGES]
    write_csvs(entities, relationships, tmp_path)
    return DuckDBGraphBackend.from_csv(
        tmp_path / "entities.csv", tmp_path / "relationships.csv", ontology
    )


def test_pattern_1_hierarchy_runs_on_tiny_domain(tmp_path):
    backend = _build_tiny_domain_backend(tmp_path)
    hierarchy = backend.get_entity_hierarchy("svc-a", direction="up")
    names = [p["name"] for p in hierarchy["parents"]]
    assert names == ["Payments Service", "Ledger Service"]


def test_pattern_2_blocker_chains_caused_by_tiny_domain(tmp_path):
    backend = _build_tiny_domain_backend(tmp_path)
    hits = backend.find_blockers("inc-1", max_depth=5)
    # max_depth clamps to caused_by's declared cap of 2
    assert [h.distance for h in hits] == [1, 2]
    assert [h.name for h in hits] == ["Middle Incident", "Leaf Incident"]


def test_pattern_3_batch_enrichment_tiny_domain(tmp_path):
    backend = _build_tiny_domain_backend(tmp_path)
    enriched = backend.enrich_entities_batch(["svc-a", "inc-1"])

    assert set(enriched.keys()) == {"svc-a", "inc-1"}
    assert [h.name for h in enriched["svc-a"].hierarchy] == ["Payments Service", "Ledger Service"]
    assert [o.name for o in enriched["svc-a"].owners] == ["Platform Team"]
    assert {b.name for b in enriched["inc-1"].blockers} == {"Middle Incident", "Leaf Incident"}


# -- Schema portability: the same vocabulary over a different physical schema ----------------

def _build_alt_schema_backend() -> DuckDBGraphBackend:
    """Build the graph as the README teaches: two views, four node columns, nothing else.

    Created with explicit DDL rather than `from_csv` precisely because `from_csv` would load
    the generator's own wide column set — the point here is a node table that has ONLY the
    four columns `table_config` declares as required.
    """
    ontology = Ontology.from_source(FileOntologySource(ALT_SCHEMA_PATH))
    backend = DuckDBGraphBackend(":memory:", ontology)
    backend.conn.execute(
        "CREATE TABLE graph_nodes (node_id VARCHAR, title VARCHAR, node_type VARCHAR, state VARCHAR)"
    )
    backend.conn.execute(
        "CREATE TABLE graph_edges (src_id VARCHAR, dst_id VARCHAR, edge_type VARCHAR)"
    )
    backend.conn.executemany(
        "INSERT INTO graph_nodes VALUES (?, ?, ?, ?)", [list(n) for n in _NODES]
    )
    backend.conn.executemany(
        "INSERT INTO graph_edges VALUES (?, ?, ?)", [list(e) for e in _EDGES]
    )
    return backend


def _run_every_backend_method(backend) -> dict[str, object]:
    """Call every method on the `GraphBackend` Protocol and return comparable results.

    Deliberately exercises all eight rather than only the three that go through
    `patterns/`: the column-name assumptions that broke schema portability lived in the five
    backend-specific queries, which no conformance test previously touched at all.
    """
    return {
        "get_entity_hierarchy": backend.get_entity_hierarchy("svc-a", direction="both"),
        "find_blockers": [h.model_dump() for h in backend.find_blockers("inc-1", max_depth=5)],
        "find_risks_for_entity": [
            r.model_dump() for r in backend.find_risks_for_entity("inc-1")
        ],
        "get_entity_owners": [o.model_dump() for o in backend.get_entity_owners("svc-a")],
        "traverse_relationships": backend.traverse_relationships(
            "svc-a", "runs_on", depth=2, direction="out"
        ),
        "enrich_entities_batch": {
            k: v.model_dump()
            for k, v in backend.enrich_entities_batch(["svc-a", "inc-1"]).items()
        },
        "find_by_name": backend.find_by_name("Checkout", exact=False).model_dump(),
        "get_descendant_counts": backend.get_descendant_counts("Service"),
    }


def test_every_backend_method_runs_on_a_four_column_node_schema(tmp_path):
    """All eight GraphBackend methods, on a node table with only the four required columns.

    Before `TableConfig.node_extra_columns` existed, seven of these eight raised
    `BinderException` on a missing `owner_id` / `description` / `properties` column, or on
    `struct_agg` emitting canonical names against physical ones.
    """
    reference = _run_every_backend_method(_build_tiny_domain_backend(tmp_path))
    alt = _run_every_backend_method(_build_alt_schema_backend())

    assert set(alt) == set(reference)
    for method, expected in reference.items():
        assert alt[method] == expected, f"{method} differs between the two schemas"


def test_alt_schema_results_are_not_vacuous(tmp_path):
    """Guard the comparison above: if both schemas returned nothing it would pass trivially."""
    backend = _build_alt_schema_backend()
    assert [p["name"] for p in backend.get_entity_hierarchy("svc-a")["parents"]] == [
        "Payments Service",
        "Ledger Service",
    ]
    assert [b.name for b in backend.find_blockers("inc-1", max_depth=5)] == [
        "Middle Incident",
        "Leaf Incident",
    ]
    assert [o.name for o in backend.get_entity_owners("svc-a")] == ["Platform Team"]
    assert [r.name for r in backend.find_risks_for_entity("inc-1")] == ["Middle Incident"]
    assert backend.find_by_name("Checkout", exact=False).entity_id == "svc-a"
    assert {
        row["entity_id"]: row["counts"]["Service"] for row in backend.get_descendant_counts("Service")
    } == {"svc-a": 0, "svc-b": 1, "svc-c": 2}


def test_risk_ordering_degrades_to_name_order_without_risk_level():
    """On a four-column schema `risk_level` does not exist — ordering must degrade, not raise.

    `tiny_domain_alt_schema.yaml` declares no `node_extra_columns`, so `Entity.risk_level` is
    None on every row and the severity key is uniform. The documented fallback is name order,
    which is what makes "most severe first" a safe promise on a minimal schema rather than a
    claim that only holds when an optional column happens to be declared.
    """
    backend = _build_alt_schema_backend()
    # Two more incidents that both caused inc-1, so two "risks" come back for one entity.
    backend.conn.executemany(
        "INSERT INTO graph_nodes VALUES (?, ?, ?, ?)",
        [["inc-9", "Zulu Incident", "Incident", "open"],
         ["inc-8", "Alpha Incident", "Incident", "open"]],
    )
    backend.conn.executemany(
        "INSERT INTO graph_edges VALUES (?, ?, ?)",
        [["inc-9", "inc-1", "caused_by"], ["inc-8", "inc-1", "caused_by"]],
    )

    risks = backend.find_risks_for_entity("inc-1")
    assert all(r.risk_level is None for r in risks)
    assert [r.name for r in risks] == ["Alpha Incident", "Middle Incident", "Zulu Incident"]
