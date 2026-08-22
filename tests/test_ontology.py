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
