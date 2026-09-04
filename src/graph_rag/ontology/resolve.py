"""Resolution helpers over a loaded Ontology.

These are the seam an LLM router (a later repo) targets: callers ask for a meaning
(``"upstream"``) or a depth, and the registry — not application code — decides what that
means concretely and whether it's allowed.
"""

from __future__ import annotations

from graph_rag.models import Entity, Relationship
from graph_rag.ontology.models import Ontology


def resolve_semantic(ontology: Ontology, name: str) -> list[str]:
    """Resolve a semantic alias to its concrete relationship type names.

    Args:
        ontology: The loaded ontology.
        name: The semantic alias, e.g. ``"upstream"``.

    Returns:
        The concrete relationship type names the alias resolves to.

    Raises:
        ValueError: If no semantic with this name is declared.
    """
    for semantic in ontology.semantics:
        if semantic.name == name:
            return list(semantic.relationship_types)
    raise ValueError(f"'{name}' is not a declared semantic")


def effective_max_depth(ontology: Ontology, rel_types: list[str], requested: int) -> int:
    """Clamp a requested traversal depth to the tightest cap among the given relationship types.

    Two declarations constrain the result. A `traversal: terminal` type is not recursive at
    all — it is an enrichment edge (who owns this, what threatens it), so it clamps to a single
    hop no matter what `max_depth` says or omits. A `max_depth` clamps a transitive type's
    blast radius. Both are the registry deciding, rather than application code.

    Args:
        ontology: The loaded ontology.
        rel_types: Relationship type names involved in the traversal.
        requested: The depth the caller asked for.

    Returns:
        The minimum of `requested`, 1 for any involved terminal type, and every involved
        type's declared `max_depth` (transitive types with no cap don't constrain the result).
    """
    effective = requested
    for rel_type_name in rel_types:
        rel_def = ontology.get_relationship_type(rel_type_name)
        if rel_def.traversal == "terminal":
            effective = min(effective, 1)
        if rel_def.max_depth is not None:
            effective = min(effective, rel_def.max_depth)
    return effective


def validate_edges(
    ontology: Ontology, entities: list[Entity], relationships: list[Relationship]
) -> None:
    """Validate that every edge's type is declared and its endpoints satisfy domain/range.

    Args:
        ontology: The loaded ontology.
        entities: Nodes, used to look up each edge endpoint's entity type.
        relationships: Edges to validate.

    Raises:
        ValueError: On an undeclared relationship type, a dangling endpoint reference, or an
            endpoint entity type outside the declared source_types/target_types for that
            relationship type.
    """
    entity_type_by_id = {e.entity_id: e.type for e in entities}
    rel_names = set(ontology.relationship_type_names())

    for rel in relationships:
        if rel.relationship_type not in rel_names:
            raise ValueError(
                f"edge {rel.source_entity_id}->{rel.target_entity_id} has undeclared "
                f"relationship_type '{rel.relationship_type}'"
            )
        rel_def = ontology.get_relationship_type(rel.relationship_type)

        if rel.source_entity_id not in entity_type_by_id:
            raise ValueError(f"edge references unknown source entity '{rel.source_entity_id}'")
        if rel.target_entity_id not in entity_type_by_id:
            raise ValueError(f"edge references unknown target entity '{rel.target_entity_id}'")

        source_type = entity_type_by_id[rel.source_entity_id]
        target_type = entity_type_by_id[rel.target_entity_id]

        if source_type not in rel_def.source_types:
            raise ValueError(
                f"edge {rel.source_entity_id}->{rel.target_entity_id} of type "
                f"'{rel.relationship_type}' has source entity type "
                f"'{source_type}' not in declared source_types {rel_def.source_types}"
            )
        if target_type not in rel_def.target_types:
            raise ValueError(
                f"edge {rel.source_entity_id}->{rel.target_entity_id} of type "
                f"'{rel.relationship_type}' has target entity type "
                f"'{target_type}' not in declared target_types {rel_def.target_types}"
            )
