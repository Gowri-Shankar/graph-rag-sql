"""`GraphBackend`: the Protocol every storage engine implementation satisfies.

A backend owns the parts a pattern module deliberately does not: resolving ontology semantics
to concrete relationship-type lists, clamping requested depths against the ontology's caps,
canonicalizing edge direction, and dispatching to `graph_rag.patterns` render functions with
the right dialect. Swapping DuckDB for BigQuery (or Postgres, later) is choosing a different
`GraphBackend` implementation — no other code changes.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from graph_rag.models import BlockerHit, EnrichmentResult, Entity


@runtime_checkable
class GraphBackend(Protocol):
    """Graph read operations, backed by whatever storage engine implements this Protocol."""

    def get_entity_hierarchy(self, entity_id: str, direction: str = "both") -> dict:
        """Return `{"entity": Entity | None, "parents": list[Entity], "children": list[Entity]}`."""
        ...

    def find_blockers(self, entity_id: str, max_depth: int = 3) -> list[BlockerHit]:
        """Find entities that transitively block/depend-on-block `entity_id`."""
        ...

    def find_risks_for_entity(self, entity_id: str) -> list[Entity]:
        """Find risks that threaten `entity_id` or any of its descendants."""
        ...

    def get_entity_owners(self, entity_id: str) -> list[Entity]:
        """Find people who own or are accountable for `entity_id`."""
        ...

    def traverse_relationships(
        self,
        start_id: str,
        rel_type: str | list[str],
        depth: int = 1,
        direction: str = "out",
    ) -> list[dict]:
        """Traverse relationship(s) from `start_id`; raises ValueError for multi-hop "both"."""
        ...

    def enrich_entities_batch(self, entity_ids: list[str]) -> dict[str, EnrichmentResult]:
        """Enrich every id in `entity_ids` (hierarchy, blockers, risks, owners) in one query."""
        ...

    def find_by_name(
        self, name: str, entity_type: str | None = None, exact: bool = False
    ) -> Entity | None:
        """Find a single entity by name, exact or partial (case-insensitive) match."""
        ...

    def get_goals_status_summary(self) -> list[dict]:
        """Return per-goal status plus transitive initiative/project/task counts."""
        ...
