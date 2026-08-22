"""Tests for DuckDBGraphBackend's ported methods against the tiny hand-written graph fixture."""

from __future__ import annotations

import pytest


def test_get_entity_hierarchy_returns_entity_itself(tiny_graph_backend):
    result = tiny_graph_backend.get_entity_hierarchy("proj-1", direction="both")
    assert result["entity"]["name"] == "Project One"


def test_get_entity_hierarchy_unknown_entity_returns_none(tiny_graph_backend):
    result = tiny_graph_backend.get_entity_hierarchy("does-not-exist")
    assert result["entity"] is None
    assert result["parents"] == []
    assert result["children"] == []


def test_get_entity_owners_returns_person(tiny_graph_backend):
    owners = tiny_graph_backend.get_entity_owners("proj-1")
    assert [o.name for o in owners] == ["Person One"]


def test_find_risks_for_entity_direct_hit(tiny_graph_backend):
    risks = tiny_graph_backend.find_risks_for_entity("proj-1")
    assert [r.name for r in risks] == ["Risk One"]


def test_find_risks_for_entity_via_descendant(tiny_graph_backend):
    # risk-1 threatens proj-1; task-1 belongs_to proj-1, so a risk lookup anchored above
    # task-1's ancestor (proj-1) should still surface it when walking from the goal down.
    risks = tiny_graph_backend.find_risks_for_entity("init-1")
    assert [r.name for r in risks] == ["Risk One"]


def test_traverse_relationships_single_hop_out(tiny_graph_backend):
    rows = tiny_graph_backend.traverse_relationships("task-1", "belongs_to", depth=1, direction="out")
    assert [r["name"] for r in rows] == ["Project One"]


def test_traverse_relationships_multi_hop_out(tiny_graph_backend):
    rows = tiny_graph_backend.traverse_relationships("task-1", "belongs_to", depth=3, direction="out")
    names_by_depth = {r["depth"]: r["name"] for r in rows}
    assert names_by_depth[1] == "Project One"
    assert names_by_depth[2] == "Initiative One"
    assert names_by_depth[3] == "Goal One"


def test_traverse_relationships_multi_hop_both_raises(tiny_graph_backend):
    with pytest.raises(ValueError, match="both"):
        tiny_graph_backend.traverse_relationships("task-1", "belongs_to", depth=2, direction="both")


def test_find_by_name_exact_match(tiny_graph_backend):
    entity = tiny_graph_backend.find_by_name("Project One", exact=True)
    assert entity is not None
    assert entity.entity_id == "proj-1"


def test_find_by_name_partial_match(tiny_graph_backend):
    entity = tiny_graph_backend.find_by_name("proj", exact=False)
    assert entity is not None
    assert entity.entity_id == "proj-1"


def test_find_by_name_no_match_returns_none(tiny_graph_backend):
    assert tiny_graph_backend.find_by_name("nonexistent-xyz", exact=True) is None


def test_get_goals_status_summary_counts_transitive_children(tiny_graph_backend):
    summary = tiny_graph_backend.get_goals_status_summary()
    assert len(summary) == 1
    goal = summary[0]
    assert goal["name"] == "Goal One"
    assert goal["initiative_count"] == 1
    assert goal["project_count"] == 1
    assert goal["task_count"] == 4
