"""Tests for `GraphRetriever`: filter mapping, direction inversion, and a real DuckDB run."""

from __future__ import annotations

import pytest

from graph_rag.models import Entity, GraphFilters
from graph_rag.retriever import GraphRetriever


class StubBackend:
    """Records every call it receives, standing in for a real `GraphBackend`."""

    def __init__(self, name_lookup: Entity | None = None) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []
        self._name_lookup = name_lookup

    def _record(self, name: str, *args, **kwargs) -> None:
        self.calls.append((name, args, kwargs))

    def get_entity_hierarchy(self, entity_id, direction="both"):
        self._record("get_entity_hierarchy", entity_id, direction=direction)
        return {"entity": None, "parents": [], "children": []}

    def find_blockers(self, entity_id, max_depth=3):
        self._record("find_blockers", entity_id, max_depth=max_depth)
        return []

    def find_risks_for_entity(self, entity_id):
        self._record("find_risks_for_entity", entity_id)
        return []

    def get_entity_owners(self, entity_id):
        self._record("get_entity_owners", entity_id)
        return []

    def traverse_relationships(self, start_id, rel_type, depth=1, direction="out"):
        self._record("traverse_relationships", start_id, rel_type, depth=depth, direction=direction)
        return []

    def enrich_entities_batch(self, entity_ids):
        self._record("enrich_entities_batch", entity_ids)
        return {}

    def find_by_name(self, name, entity_type=None, exact=False):
        self._record("find_by_name", name, entity_type=entity_type, exact=exact)
        return self._name_lookup

    def get_descendant_counts(self, root_type, count_types=None):
        self._record("get_descendant_counts", root_type, count_types)
        return []


def _entity(entity_id: str, name: str, type_: str = "Person") -> Entity:
    from datetime import datetime

    return Entity(entity_id=entity_id, name=name, type=type_, created_at=datetime(2025, 1, 1))


def test_get_hierarchy_delegates():
    backend = StubBackend()
    GraphRetriever(backend).get_hierarchy("proj-1", direction="up")
    assert backend.calls == [("get_entity_hierarchy", ("proj-1",), {"direction": "up"})]


def test_find_blockers_delegates():
    backend = StubBackend()
    GraphRetriever(backend).find_blockers("proj-1", max_depth=4)
    assert backend.calls == [("find_blockers", ("proj-1",), {"max_depth": 4})]


def test_traverse_by_entity_id_no_inversion():
    backend = StubBackend()
    filters = GraphFilters(entity_id="proj-1", rel_type="owns", rel_direction="in", rel_max_depth=2)
    GraphRetriever(backend).traverse(filters)
    assert backend.calls == [
        ("traverse_relationships", ("proj-1", "owns"), {"depth": 2, "direction": "in"})
    ]


def test_traverse_by_name_inverts_direction():
    """Resolving "projects OF Alex Carter, CEO of Acme Analytics" asks from the person's side,
    so the retriever must invert the analyzer's "in" direction to "out" once it resolves the
    name to an entity_id — see the module docstring for the full worked example.
    """
    backend = StubBackend(name_lookup=_entity("person-1", "Alex Carter"))
    filters = GraphFilters(name="Alex Carter", rel_type="owns", rel_direction="in")
    GraphRetriever(backend).traverse(filters)

    lookup_call, traverse_call = backend.calls
    assert lookup_call[0] == "find_by_name"
    assert traverse_call == (
        "traverse_relationships", ("person-1", "owns"), {"depth": 3, "direction": "out"}
    )


def test_traverse_by_name_no_match_returns_empty():
    backend = StubBackend(name_lookup=None)
    filters = GraphFilters(name="Nobody", rel_type="owns")
    assert GraphRetriever(backend).traverse(filters) == []


def test_traverse_requires_rel_type():
    backend = StubBackend()
    with pytest.raises(ValueError, match="rel_type"):
        GraphRetriever(backend).traverse(GraphFilters(entity_id="proj-1"))


def test_traverse_requires_entity_id_or_name():
    backend = StubBackend()
    with pytest.raises(ValueError, match="entity_id or filters.name"):
        GraphRetriever(backend).traverse(GraphFilters(rel_type="owns"))


def test_enrich_batch_delegates():
    backend = StubBackend()
    GraphRetriever(backend).enrich_batch(["a", "b"])
    assert backend.calls == [("enrich_entities_batch", (["a", "b"],), {})]


def test_find_by_name_delegates():
    backend = StubBackend()
    GraphRetriever(backend).find_by_name("Atlas", entity_type="Project", exact=True)
    assert backend.calls == [
        ("find_by_name", ("Atlas",), {"entity_type": "Project", "exact": True})
    ]


def test_descendant_counts_delegates():
    backend = StubBackend()
    GraphRetriever(backend).descendant_counts("Goal")
    assert backend.calls == [("get_descendant_counts", ("Goal", None), {})]


# -- Runs identically over the real DuckDB backend ------------------------------------------


def test_retriever_over_duckdb_backend_finds_blockers(tiny_graph_backend):
    retriever = GraphRetriever(tiny_graph_backend)
    hits = retriever.find_blockers("task-1", max_depth=3)
    assert [h.entity_id for h in hits] == ["task-2", "task-3", "task-4"]


def test_retriever_over_duckdb_backend_traverse_by_name(tiny_graph_backend):
    retriever = GraphRetriever(tiny_graph_backend)
    filters = GraphFilters(name="Person One", rel_type="owns", rel_direction="in", rel_max_depth=1)
    rows = retriever.traverse(filters)
    assert [r["name"] for r in rows] == ["Project One"]


def test_retriever_over_duckdb_backend_owners(tiny_graph_backend):
    retriever = GraphRetriever(tiny_graph_backend)
    owners = retriever.get_owners("proj-1")
    assert [o.name for o in owners] == ["Person One", "Person Two"]
