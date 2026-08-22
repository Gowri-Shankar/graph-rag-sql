"""THE PORTABILITY PROOF.

Runs all three patterns against a second ontology (Service/Incident/Team) that shares no type
names with the org graph. This must pass with ZERO changes to any file under
`src/graph_rag/patterns/` — if it doesn't, the patterns still carry domain coupling; the fix
belongs in the patterns, never in this test.
"""

from __future__ import annotations

from datetime import datetime

from graph_rag.backends.duckdb_backend import DuckDBGraphBackend
from graph_rag.generator import write_csvs
from graph_rag.models import Entity, Relationship
from graph_rag.ontology import FileOntologySource, Ontology

TINY_DOMAIN_PATH = "tests/fixtures/tiny_domain.yaml"


def _entity(entity_id: str, name: str, type_: str) -> Entity:
    return Entity(entity_id=entity_id, name=name, type=type_, created_at=datetime(2025, 1, 1))


def _edge(source: str, target: str, rel_type: str) -> Relationship:
    return Relationship(
        source_entity_id=source,
        target_entity_id=target,
        relationship_type=rel_type,
        created_at=datetime(2025, 1, 1),
    )


def _build_tiny_domain_backend(tmp_path) -> DuckDBGraphBackend:
    ontology = Ontology.from_source(FileOntologySource(TINY_DOMAIN_PATH))
    entities = [
        _entity("svc-a", "Checkout Service", "Service"),
        _entity("svc-b", "Payments Service", "Service"),
        _entity("svc-c", "Ledger Service", "Service"),
        _entity("team-1", "Platform Team", "Team"),
        _entity("inc-1", "Root Incident", "Incident"),
        _entity("inc-2", "Middle Incident", "Incident"),
        _entity("inc-3", "Leaf Incident", "Incident"),
    ]
    relationships = [
        _edge("svc-a", "svc-b", "runs_on"),
        _edge("svc-b", "svc-c", "runs_on"),
        _edge("inc-3", "inc-2", "caused_by"),
        _edge("inc-2", "inc-1", "caused_by"),
        _edge("team-1", "svc-a", "on_call_for"),
    ]
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
