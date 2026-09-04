"""`GraphRetriever`: the facade a caller (e.g. a RAG agent) targets instead of a `GraphBackend`.

Maps a `GraphFilters` object onto the right backend call so callers never build SQL and never
see per-backend argument shapes. `GraphRetriever` is backend-agnostic by construction — it is
constructed with a `GraphBackend`, and every method is pure delegation plus the one piece of
routing logic below.

Direction-inversion note: when a caller resolves a starting entity by NAME rather than by ID
(e.g. "projects OF Alex Carter, CEO of Acme Analytics" — asked from the person's side of an
ownership-shaped relationship), the requested `rel_direction` was chosen from the *other* end's
point of view and must be flipped once the name resolves to an entity_id. This is unrelated to
edge-direction canonicalization: that happens once, at load time, in each `GraphBackend` (see
`backends/duckdb_backend.py`), so pattern SQL only ever walks a canonical direction. This is a
per-request routing concern — which way THIS caller's question points relative to the anchor it
just resolved — and belongs here, in the facade, not in a backend or a pattern.
"""

from __future__ import annotations

from graph_rag.backends.base import GraphBackend
from graph_rag.models import BlockerHit, EnrichmentResult, Entity, GraphFilters

_INVERSE_DIRECTION = {"in": "out", "out": "in", "both": "both"}


class GraphRetriever:
    """Routes `GraphFilters`-described requests to the right `GraphBackend` call."""

    def __init__(self, backend: GraphBackend) -> None:
        self.backend = backend

    def get_hierarchy(self, entity_id: str, direction: str = "both") -> dict:
        """Return `{"entity", "parents", "children"}` for `entity_id`."""
        return self.backend.get_entity_hierarchy(entity_id, direction)

    def find_blockers(self, entity_id: str, max_depth: int = 3) -> list[BlockerHit]:
        """Find entities that transitively block/depend-on-block `entity_id`."""
        return self.backend.find_blockers(entity_id, max_depth)

    def find_risks(self, entity_id: str) -> list[Entity]:
        """Find risks that threaten `entity_id` or any of its descendants."""
        return self.backend.find_risks_for_entity(entity_id)

    def get_owners(self, entity_id: str) -> list[Entity]:
        """Find people who own or are accountable for `entity_id`."""
        return self.backend.get_entity_owners(entity_id)

    def traverse(self, filters: GraphFilters) -> list[dict]:
        """Traverse relationship(s) described by `filters`.

        Requires `filters.rel_type` and either `filters.entity_id` or `filters.name`. If only
        `name` is given, resolves it to an entity first and inverts `rel_direction` — see the
        module docstring.
        """
        if not filters.rel_type:
            raise ValueError("filters.rel_type is required for traverse()")

        entity_id = filters.entity_id
        direction = filters.rel_direction

        if entity_id is None and filters.name:
            entity = self.backend.find_by_name(
                filters.name,
                filters.entity_type,
                exact=(filters.name_match == "exact"),
            )
            if entity is None:
                return []
            entity_id = entity.entity_id
            direction = _INVERSE_DIRECTION[direction]

        if entity_id is None:
            raise ValueError("filters.entity_id or filters.name is required for traverse()")

        return self.backend.traverse_relationships(
            entity_id, filters.rel_type, depth=filters.rel_max_depth, direction=direction
        )

    def enrich_batch(self, entity_ids: list[str]) -> dict[str, EnrichmentResult]:
        """Enrich every id in `entity_ids` in one query."""
        return self.backend.enrich_entities_batch(entity_ids)

    def find_by_name(
        self, name: str, entity_type: str | None = None, exact: bool = False
    ) -> Entity | None:
        """Find a single entity by name, exact or partial (case-insensitive) match."""
        return self.backend.find_by_name(name, entity_type, exact=exact)

    def descendant_counts(
        self, root_type: str, count_types: list[str] | None = None
    ) -> list[dict]:
        """Per-root columns plus transitive descendant counts, broken down by entity type.

        `count_types` defaults to every entity type the ontology declares.
        """
        return self.backend.get_descendant_counts(root_type, count_types)
