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


def test_find_risks_for_entity_via_transitive_descendant(tiny_graph_backend):
    """A risk on a descendant more than one hop down must surface.

    Anchored at goal-1, which is 2 hops above proj-1 (goal-1 <- init-1 <- proj-1). risk-2
    threatens init-1 — goal-1's DIRECT child, so a walk that only manages one hop still finds
    it. risk-1 threatens proj-1, two hops down, and is the one that requires the `children`
    CTE's recursive step to keep descending. Asserting both pins the transitive reach rather
    than just the base case.
    """
    risks = tiny_graph_backend.find_risks_for_entity("goal-1")
    assert sorted(r.name for r in risks) == ["Risk One", "Risk Two"]


def test_find_risks_for_entity_excludes_risk_on_an_ancestor(tiny_graph_backend):
    """A risk threatening an ANCESTOR must not be reported as threatening this entity.

    Regression guard for the more dangerous half of the descendant-walk bug: when the
    recursive step inverted direction it oscillated child -> parent -> child, so walking down
    from proj-1 reached its own parent init-1 and wrongly attributed init-1's risk-2 to
    proj-1. proj-1's only true descendants here are task-1..task-4, none of which is
    threatened, so risk-1 (a direct hit on proj-1 itself) must be the only result.
    """
    risks = tiny_graph_backend.find_risks_for_entity("proj-1")
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


def test_get_descendant_counts_counts_transitive_children(tiny_graph_backend):
    summary = tiny_graph_backend.get_descendant_counts("Goal", ["Initiative", "Project", "Task"])
    assert len(summary) == 1
    goal = summary[0]
    assert goal["name"] == "Goal One"
    assert goal["counts"] == {"Initiative": 1, "Project": 1, "Task": 4}


def test_get_descendant_counts_defaults_count_types_to_the_whole_ontology(tiny_graph_backend):
    """With no `count_types`, every declared entity type appears — zero-filled if absent."""
    goal = tiny_graph_backend.get_descendant_counts("Goal")[0]
    assert goal["counts"] == {
        "Goal": 0, "Initiative": 1, "Project": 1, "Task": 4, "Person": 0, "Risk": 0
    }
