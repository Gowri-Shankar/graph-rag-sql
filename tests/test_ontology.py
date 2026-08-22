"""Tests for the ontology registry: loading, resolution, and validation of malformed registries."""

from __future__ import annotations

import copy

import pytest
import yaml

from graph_rag.ontology import (
    FileOntologySource,
    Ontology,
    effective_max_depth,
    resolve_semantic,
)

ORG_GRAPH_PATH = "ontology/org_graph.yaml"


@pytest.fixture
def raw_org_graph() -> dict:
    with open(ORG_GRAPH_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_org_graph_loads_and_validates():
    ontology = Ontology.from_source(FileOntologySource(ORG_GRAPH_PATH))
    assert "Goal" in ontology.entity_type_names()
    assert "Task" in ontology.entity_type_names()
    assert "blocks" in ontology.relationship_type_names()
    assert "belongs_to" in ontology.relationship_type_names()


def test_resolve_semantic_upstream():
    ontology = Ontology.from_source(FileOntologySource(ORG_GRAPH_PATH))
    assert resolve_semantic(ontology, "upstream") == ["blocks", "depends_on"]


def test_resolve_semantic_unknown_raises():
    ontology = Ontology.from_source(FileOntologySource(ORG_GRAPH_PATH))
    with pytest.raises(ValueError):
        resolve_semantic(ontology, "not_a_real_semantic")


def test_effective_max_depth_clamps_to_registry_cap():
    ontology = Ontology.from_source(FileOntologySource(ORG_GRAPH_PATH))
    depth = effective_max_depth(ontology, ["blocks", "depends_on"], requested=10)
    assert depth == 5


def test_every_type_has_a_description(raw_org_graph):
    for entity_type in raw_org_graph["entity_types"]:
        assert entity_type["description"].strip()
    for rel_type in raw_org_graph["relationship_types"]:
        assert rel_type["description"].strip()


def test_semantic_naming_undeclared_relationship_type_raises(raw_org_graph):
    bad = copy.deepcopy(raw_org_graph)
    bad["semantics"].append({"name": "bogus", "relationship_types": ["not_a_real_type"]})
    with pytest.raises(ValueError):
        Ontology.model_validate(bad)


def test_relationship_with_undeclared_domain_type_raises(raw_org_graph):
    bad = copy.deepcopy(raw_org_graph)
    bad["relationship_types"][0]["source_types"] = ["NotAnEntityType"]
    with pytest.raises(ValueError):
        Ontology.model_validate(bad)


def test_dangling_inverse_raises(raw_org_graph):
    bad = copy.deepcopy(raw_org_graph)
    bad["relationship_types"][0]["inverse"] = "not_a_real_relationship_type"
    with pytest.raises(ValueError):
        Ontology.model_validate(bad)


def test_max_depth_zero_raises(raw_org_graph):
    bad = copy.deepcopy(raw_org_graph)
    bad["relationship_types"][0]["max_depth"] = 0
    with pytest.raises(ValueError):
        Ontology.model_validate(bad)
