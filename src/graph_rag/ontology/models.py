"""Pydantic models for the ontology registry.

The registry is a small, bounded, plain-JSON-serializable property-graph vocabulary — not an
RDF/OWL/SHACL document. That keeps validation cheap and keeps a future export to a standard
format a small, optional add-on rather than a rewrite.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, field_validator, model_validator

if TYPE_CHECKING:
    from graph_rag.ontology.sources import OntologySource


class EntityTypeDef(BaseModel):
    """A declared node type, e.g. ``Project`` or ``Person``."""

    name: str
    description: str


class RelationshipTypeDef(BaseModel):
    """A declared edge type, with its domain/range, direction, and traversal behavior."""

    name: str
    description: str
    source_types: list[str]
    target_types: list[str]
    inverse: str | None = None
    traversal: Literal["transitive", "terminal"]
    canonical_direction: Literal["source_to_target", "target_to_source"]
    max_depth: int | None = None
    fan_out_limit: int | None = None

    @field_validator("max_depth")
    @classmethod
    def _max_depth_positive(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError(f"max_depth must be >= 1, got {value}")
        return value


class Semantic(BaseModel):
    """A caller-facing alias mapping a meaning (e.g. ``upstream``) to concrete relationship types."""

    name: str
    relationship_types: list[str]


class TableConfig(BaseModel):
    """Table and column names backing the graph, so none of these are literals in a query."""

    node_table: str
    edge_table: str
    node_id_column: str
    node_name_column: str
    node_type_column: str
    node_status_column: str
    edge_source_column: str
    edge_target_column: str
    edge_type_column: str


class Ontology(BaseModel):
    """A fully hydrated, validated graph vocabulary: entity types, relationship types, aliases."""

    name: str
    version: str
    entity_types: list[EntityTypeDef]
    relationship_types: list[RelationshipTypeDef]
    semantics: list[Semantic]
    table_config: TableConfig

    @model_validator(mode="after")
    def _validate_cross_references(self) -> Ontology:
        entity_names = {et.name for et in self.entity_types}
        rel_names = {rt.name for rt in self.relationship_types}

        for rel in self.relationship_types:
            for type_name in rel.source_types:
                if type_name not in entity_names:
                    raise ValueError(
                        f"relationship type '{rel.name}' has source_types entry "
                        f"'{type_name}' which is not a declared entity type"
                    )
            for type_name in rel.target_types:
                if type_name not in entity_names:
                    raise ValueError(
                        f"relationship type '{rel.name}' has target_types entry "
                        f"'{type_name}' which is not a declared entity type"
                    )
            if rel.inverse is not None and rel.inverse not in rel_names:
                raise ValueError(
                    f"relationship type '{rel.name}' declares inverse '{rel.inverse}' "
                    f"which is not a declared relationship type"
                )

        for semantic in self.semantics:
            for rel_type in semantic.relationship_types:
                if rel_type not in rel_names:
                    raise ValueError(
                        f"semantic '{semantic.name}' references relationship type "
                        f"'{rel_type}' which is not declared"
                    )

        return self

    def entity_type_names(self) -> list[str]:
        """Return all declared entity type names."""
        return [et.name for et in self.entity_types]

    def relationship_type_names(self) -> list[str]:
        """Return all declared relationship type names."""
        return [rt.name for rt in self.relationship_types]

    def get_relationship_type(self, name: str) -> RelationshipTypeDef:
        """Look up a relationship type definition by name, raising if undeclared."""
        for rel in self.relationship_types:
            if rel.name == name:
                return rel
        raise ValueError(f"'{name}' is not a declared relationship type")

    @classmethod
    def from_source(cls, source: OntologySource) -> Ontology:
        """Load a fully hydrated, validated Ontology from any OntologySource."""
        return source.load()
