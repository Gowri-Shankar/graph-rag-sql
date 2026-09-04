"""Shared fixtures: a tiny, hand-written graph with known, exact-assertable shape."""

from __future__ import annotations

from datetime import datetime

import pytest

from graph_rag.backends.duckdb_backend import DuckDBGraphBackend
from graph_rag.models import Entity, Relationship
from graph_rag.ontology import FileOntologySource, Ontology

ORG_GRAPH_PATH = "ontology/org_graph.yaml"


@pytest.fixture
def org_ontology() -> Ontology:
    return Ontology.from_source(FileOntologySource(ORG_GRAPH_PATH))


def _entity(entity_id: str, name: str, type_: str, status: str = "in_progress") -> Entity:
    return Entity(
        entity_id=entity_id,
        name=name,
        type=type_,
        status=status,
        created_at=datetime(2025, 1, 1),
    )


def _edge(source: str, target: str, rel_type: str) -> Relationship:
    return Relationship(
        source_entity_id=source,
        target_entity_id=target,
        relationship_type=rel_type,
        created_at=datetime(2025, 1, 1),
    )


@pytest.fixture
def tiny_graph_backend(org_ontology: Ontology, tmp_path) -> DuckDBGraphBackend:
    """A hand-written ~12-node graph with a known hierarchy, a known 3-hop blocker chain,
    two risks at different hierarchy levels, and one owner — small enough that every
    assertion in test_patterns.py and test_duckdb_backend.py is exact.

    Shape:
        goal-1 <- init-1 <- proj-1 <- task-1 <- task-2 <- task-3 <- task-4  (blocks chain,
                                                                              task-4 blocks
                                                                              task-3 blocks
                                                                              task-2 blocks
                                                                              task-1)
        person-1 --owns--> proj-1
        person-2 --accountable_for--> goal-1
        risk-1 --threatens--> proj-1
        risk-2 --threatens--> init-1

    The two risks sit at deliberately different depths so descendant-walk direction is
    testable in both directions of error: from goal-1, risk-1 is a 2-hop descendant hit
    (a walk that only reaches direct children misses it), while from proj-1, risk-2 sits on
    an ANCESTOR and must not appear (a walk that oscillates child->parent wrongly finds it).
    """
    entities = [
        _entity("goal-1", "Goal One", "Goal"),
        _entity("init-1", "Initiative One", "Initiative"),
        _entity("proj-1", "Project One", "Project", status="at_risk"),
        _entity("task-1", "Task One", "Task", status="blocked"),
        _entity("task-2", "Task Two", "Task"),
        _entity("task-3", "Task Three", "Task"),
        _entity("task-4", "Task Four", "Task"),
        _entity("person-1", "Person One", "Person", status="active"),
        _entity("person-2", "Person Two", "Person", status="active"),
        _entity("risk-1", "Risk One", "Risk", status="open"),
        _entity("risk-2", "Risk Two", "Risk", status="open"),
    ]
    relationships = [
        _edge("init-1", "goal-1", "belongs_to"),
        _edge("proj-1", "init-1", "belongs_to"),
        _edge("task-1", "proj-1", "belongs_to"),
        _edge("task-2", "proj-1", "belongs_to"),
        _edge("task-3", "proj-1", "belongs_to"),
        _edge("task-4", "proj-1", "belongs_to"),
        _edge("task-2", "task-1", "blocks"),
        _edge("task-3", "task-2", "blocks"),
        _edge("task-4", "task-3", "blocks"),
        _edge("person-1", "proj-1", "owns"),
        _edge("person-2", "goal-1", "accountable_for"),
        _edge("risk-1", "proj-1", "threatens"),
        _edge("risk-2", "init-1", "threatens"),
    ]

    entities_csv = tmp_path / "entities.csv"
    relationships_csv = tmp_path / "relationships.csv"

    from graph_rag.generator import write_csvs

    write_csvs(entities, relationships, tmp_path)

    return DuckDBGraphBackend.from_csv(entities_csv, relationships_csv, org_ontology)
