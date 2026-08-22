"""The ontology registry: models, sources, and resolution helpers."""

from graph_rag.ontology.models import (
    EntityTypeDef,
    Ontology,
    RelationshipTypeDef,
    Semantic,
    TableConfig,
)
from graph_rag.ontology.resolve import effective_max_depth, resolve_semantic, validate_edges
from graph_rag.ontology.sources import FileOntologySource, OntologySource, TableOntologySource

__all__ = [
    "EntityTypeDef",
    "FileOntologySource",
    "Ontology",
    "OntologySource",
    "RelationshipTypeDef",
    "Semantic",
    "TableConfig",
    "TableOntologySource",
    "effective_max_depth",
    "resolve_semantic",
    "validate_edges",
]
