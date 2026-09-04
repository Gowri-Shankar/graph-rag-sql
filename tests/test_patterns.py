"""Tests for the three pattern modules, run against the tiny hand-written graph fixture."""

from __future__ import annotations


def test_find_blockers_respects_depth_bound(tiny_graph_backend):
    hits = tiny_graph_backend.find_blockers("task-1", max_depth=2)
    assert [h.distance for h in hits] == [1, 2]
    assert [h.name for h in hits] == ["Task Two", "Task Three"]


def test_find_blockers_full_chain(tiny_graph_backend):
    hits = tiny_graph_backend.find_blockers("task-1", max_depth=5)
    assert [h.distance for h in hits] == [1, 2, 3]
    assert [h.name for h in hits] == ["Task Two", "Task Three", "Task Four"]


def test_find_blockers_rel_chain_and_name_chain_are_exact(tiny_graph_backend):
    hits = tiny_graph_backend.find_blockers("task-1", max_depth=5)
    third = hits[-1]
    assert third.rel_chain == ["blocks", "blocks", "blocks"]
    assert third.name_chain == ["Task Two", "Task Three", "Task Four"]


def test_hierarchy_up_from_task_reaches_goal(tiny_graph_backend):
    hierarchy = tiny_graph_backend.get_entity_hierarchy("task-1", direction="up")
    names_by_depth = {p["depth"]: p["name"] for p in hierarchy["parents"]}
    assert names_by_depth[1] == "Project One"
    assert names_by_depth[2] == "Initiative One"
    assert names_by_depth[3] == "Goal One"


def test_hierarchy_down_from_goal_reaches_project(tiny_graph_backend):
    hierarchy = tiny_graph_backend.get_entity_hierarchy("goal-1", direction="down")
    names = {c["name"] for c in hierarchy["children"]}
    assert "Initiative One" in names
    assert "Project One" in names


def test_enrichment_returns_keys_for_every_requested_id(tiny_graph_backend):
    enriched = tiny_graph_backend.enrich_entities_batch(["proj-1", "task-4"])
    assert set(enriched.keys()) == {"proj-1", "task-4"}


def test_enrichment_empty_groups_are_empty_lists_not_none(tiny_graph_backend):
    enriched = tiny_graph_backend.enrich_entities_batch(["task-4"])
    result = enriched["task-4"]
    assert result.risks == []
    assert result.owners == []


def test_enrichment_matches_standalone_lookups(tiny_graph_backend):
    enriched = tiny_graph_backend.enrich_entities_batch(["proj-1"])
    result = enriched["proj-1"]

    standalone_owners = {o.name for o in tiny_graph_backend.get_entity_owners("proj-1")}
    standalone_risks = {r.name for r in tiny_graph_backend.find_risks_for_entity("proj-1")}

    assert {o.name for o in result.owners} == standalone_owners == {"Person One", "Person Two"}
    assert {r.name for r in result.risks} == standalone_risks == {"Risk One"}
