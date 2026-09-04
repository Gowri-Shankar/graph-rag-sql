"""`GraphBackend`: the Protocol every storage engine implementation satisfies.

A backend owns the parts a pattern module deliberately does not: resolving ontology semantics
to concrete relationship-type lists, clamping requested depths against the ontology's caps,
canonicalizing edge direction, and dispatching to `graph_rag.patterns` render functions with
the right dialect. Swapping DuckDB for BigQuery (or Postgres, later) is choosing a different
`GraphBackend` implementation — no other code changes.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from graph_rag.models import BlockerHit, EnrichmentResult, Entity

# The column a descendant-count query names its per-row entity type. Kept here rather than in
# either backend so the two render the same shape and share the pivot below.
DESCENDANT_TYPE_COLUMN = "descendant_type"
DESCENDANT_COUNT_COLUMN = "descendant_count"


def pivot_descendant_counts(rows: list[dict[str, Any]], count_types: list[str]) -> list[dict]:
    """Fold `(root, descendant_type, n)` rows into one row per root with a `counts` dict.

    Counting per type in SQL and pivoting here keeps the query free of entity-type literals:
    a `CASE WHEN type = 'Project'` pivot needs one literal per counted type baked into the
    SELECT list, which is exactly the coupling the ontology registry exists to remove. Every
    name in `count_types` appears in `counts`, zero-filled, so a caller never has to
    distinguish "no descendants of that type" from "column absent".
    """
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        row = dict(row)
        descendant_type = row.pop(DESCENDANT_TYPE_COLUMN, None)
        count = row.pop(DESCENDANT_COUNT_COLUMN, 0)
        entry = grouped.setdefault(
            row["entity_id"], {**row, "counts": dict.fromkeys(count_types, 0)}
        )
        if descendant_type in entry["counts"]:
            entry["counts"][descendant_type] = int(count)
    return list(grouped.values())


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

    def get_descendant_counts(
        self, root_type: str, count_types: list[str] | None = None
    ) -> list[dict]:
        """Per-root transitive descendant counts, broken down by descendant entity type.

        Args:
            root_type: The entity type to report on, one row per entity of that type.
            count_types: Descendant entity types to count. Defaults to every type the
                ontology declares.

        Returns:
            One dict per root: the root's own projected columns plus `counts`, a
            `{entity_type: int}` mapping with every name in `count_types` present.
        """
        ...
