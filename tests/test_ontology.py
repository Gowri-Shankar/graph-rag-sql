"""Tests for the ontology registry: loading, resolution, and validation of malformed registries."""

from __future__ import annotations

import copy
import json

import duckdb
import pytest
import yaml

from graph_rag.dialects.duckdb import DuckDbDialect
from graph_rag.ontology import (
    FileOntologySource,
    Ontology,
    TableOntologySource,
    effective_max_depth,
    resolve_semantic,
)

ORG_GRAPH_PATH = "ontology/org_graph.yaml"
TINY_DOMAIN_PATH = "tests/fixtures/tiny_domain.yaml"


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


def test_effective_max_depth_clamps_terminal_types_to_one_hop(raw_org_graph):
    """A `traversal: terminal` type is enrichment-only and never recurses.

    The registry field used to be inert — declared, documented, and read by no code, so the
    README's claim that it governs recursion was untrue. Uncapping `max_depth` here isolates
    `traversal` as the thing doing the clamping.
    """
    raw = copy.deepcopy(raw_org_graph)
    for rel in raw["relationship_types"]:
        if rel["name"] == "threatens":
            rel["max_depth"] = None
    ontology = Ontology.model_validate(raw)

    assert ontology.get_relationship_type("threatens").traversal == "terminal"
    assert effective_max_depth(ontology, ["threatens"], requested=5) == 1
    # A transitive type with no cap is still unconstrained.
    assert effective_max_depth(ontology, ["belongs_to"], requested=4) == 4


def test_node_extra_columns_defaults_to_empty():
    """The default is what makes a minimal four-column node view work with no declaration."""
    ontology = Ontology.from_source(FileOntologySource(TINY_DOMAIN_PATH))
    assert ontology.table_config.node_extra_columns == []


def test_node_projection_renders_canonical_aliases_plus_extras():
    tc = Ontology.from_source(FileOntologySource(ORG_GRAPH_PATH)).table_config
    projection = tc.node_projection("e")
    assert projection.startswith(
        "e.entity_id AS entity_id, e.name AS name, e.type AS type, e.status AS status"
    )
    assert "e.description" in projection
    assert tc.node_projection().startswith("entity_id AS entity_id")


def test_node_extra_columns_may_not_repeat_a_required_column(raw_org_graph):
    raw = copy.deepcopy(raw_org_graph)
    raw["table_config"]["node_extra_columns"] = ["name"]
    with pytest.raises(ValueError, match="projected twice"):
        Ontology.model_validate(raw)


def test_node_extra_columns_may_not_collide_with_a_canonical_alias(raw_org_graph):
    """Physical `subject` aliased to `name` collides with an extra column literally named `name`."""
    raw = copy.deepcopy(raw_org_graph)
    raw["table_config"]["node_name_column"] = "subject"
    raw["table_config"]["node_extra_columns"] = ["name"]
    with pytest.raises(ValueError, match="collides with the canonical output name"):
        Ontology.model_validate(raw)


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


def _seed_table_ontology(conn: duckdb.DuckDBPyConnection, ontology: Ontology) -> None:
    """Populate ontology_* tables in `conn` from an already-loaded Ontology — the table-backed
    counterpart to `ontology/org_graph.yaml`, used only to prove `TableOntologySource` hydrates
    the identical vocabulary a file source would.
    """
    conn.execute(
        "CREATE TABLE ontology_entity_types (ontology_name VARCHAR, seq BIGINT, name VARCHAR, description VARCHAR)"
    )
    conn.execute(
        "CREATE TABLE ontology_relationship_types ("
        "ontology_name VARCHAR, seq BIGINT, name VARCHAR, description VARCHAR, "
        "source_types VARCHAR, target_types VARCHAR, inverse VARCHAR, "
        "traversal VARCHAR, canonical_direction VARCHAR, max_depth BIGINT, fan_out_limit BIGINT)"
    )
    conn.execute(
        "CREATE TABLE ontology_semantics (ontology_name VARCHAR, seq BIGINT, name VARCHAR, relationship_types VARCHAR)"
    )

    for i, et in enumerate(ontology.entity_types):
        conn.execute(
            "INSERT INTO ontology_entity_types VALUES ($ontology_name, $seq, $name, $description)",
            {"ontology_name": ontology.name, "seq": i, "name": et.name, "description": et.description},
        )
    for i, rt in enumerate(ontology.relationship_types):
        conn.execute(
            "INSERT INTO ontology_relationship_types VALUES "
            "($ontology_name, $seq, $name, $description, $source_types, $target_types, "
            "$inverse, $traversal, $canonical_direction, $max_depth, $fan_out_limit)",
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


def _duckdb_executor(conn: duckdb.DuckDBPyConnection):
    def execute(sql: str, params: dict) -> list[dict]:
        cursor = conn.execute(sql, params)
        columns = [d[0] for d in cursor.description]
        return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]

    return execute


def test_table_ontology_source_matches_file_source():
    """The "dynamic" proof: a caller cannot tell a table-backed ontology from a file one."""
    file_ontology = Ontology.from_source(FileOntologySource(ORG_GRAPH_PATH))

    conn = duckdb.connect(":memory:")
    _seed_table_ontology(conn, file_ontology)

    table_ontology = TableOntologySource(
        execute=_duckdb_executor(conn),
        dialect=DuckDbDialect(),
        ontology_name=file_ontology.name,
        version=file_ontology.version,
        table_config=file_ontology.table_config,
    ).load()

    assert table_ontology == file_ontology
