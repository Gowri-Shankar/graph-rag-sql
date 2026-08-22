"""Pattern 1: bounded recursive hierarchy traversal.

Walks a transitive "part-of" relationship either up toward ancestors or down toward
descendants, with a hard depth bound. The bound is what makes this viable on a
warehouse at all: an unbounded recursive CTE over a cyclic or merely very deep hierarchy can
loop indefinitely or return an unbounded number of rows, and every additional hop is another
self-join over the full edge table.

Complexity: one recursive self-join of the edge table per hop, so cost scales with
(branching factor) ^ (depth), which is exactly why `effective_max_depth` clamps the caller's
requested depth against the ontology's per-type cap before this ever runs.

No domain literals: `rel_types` is a resolved list of relationship-type names (typically
`resolve_semantic(ontology, "hierarchy")`), and table/column names come from
`ontology.table_config`. Deviation from the ported source: the source additionally orders
results by a hardcoded `CASE type WHEN 'Goal' THEN 1 ...` — a literal specific to one domain's
entity types. This pattern orders by depth alone; a caller wanting type-priority display
ordering can sort the returned rows client-side using `ontology.entity_types`' declared order.
"""

from __future__ import annotations

from graph_rag.dialects.base import SqlDialect
from graph_rag.ontology.models import Ontology


def render(
    dialect: SqlDialect,
    ontology: Ontology,
    rel_types: list[str],
    entity_id: str,
    direction: str,
    max_depth: int,
) -> tuple[str, dict[str, object]]:
    """Render the recursive hierarchy query.

    Args:
        dialect: The SQL dialect to render against.
        ontology: Supplies table/column names via `ontology.table_config`.
        rel_types: Concrete, pre-resolved relationship type names to traverse.
        entity_id: The anchor entity to walk from.
        direction: ``"up"`` walks toward ancestors (following edges from child/source to
            parent/target); ``"down"`` walks toward descendants (the reverse). This follows
            the convention that a hierarchy edge points child(source) -> parent(target).
        max_depth: Depth bound, already clamped by `effective_max_depth`.

    Returns:
        `(sql, params)`. `params` has keys `"entity_id"` and `"rel_types"`.

    Raises:
        ValueError: If `direction` is not `"up"` or `"down"`.
    """
    if direction not in ("up", "down"):
        raise ValueError(f"direction must be 'up' or 'down', got {direction!r}")

    tc = ontology.table_config
    entities = dialect.qualify_table(tc.node_table)
    edges = dialect.qualify_table(tc.edge_table)
    id_col, name_col, type_col, status_col = (
        tc.node_id_column,
        tc.node_name_column,
        tc.node_type_column,
        tc.node_status_column,
    )
    edge_type_col = tc.edge_type_column

    # "up": base case joins on the target (parent) side; the recursive step keeps walking
    # via the source column. "down" is the mirror image.
    outward_col = tc.edge_target_column if direction == "up" else tc.edge_source_column
    inward_col = tc.edge_source_column if direction == "up" else tc.edge_target_column

    entity_param = dialect.param("entity_id")
    rel_membership = dialect.array_membership(f"r.{edge_type_col}", "rel_types")

    sql = f"""
        WITH RECURSIVE hierarchy AS (
            SELECT
                e.{id_col} AS entity_id, e.{name_col} AS name,
                e.{type_col} AS type, e.{status_col} AS status,
                1 AS depth
            FROM {edges} r
            JOIN {entities} e ON r.{outward_col} = e.{id_col}
            WHERE r.{inward_col} = {entity_param}
                AND {rel_membership}

            UNION ALL

            SELECT
                e.{id_col} AS entity_id, e.{name_col} AS name,
                e.{type_col} AS type, e.{status_col} AS status,
                h.depth + 1
            FROM hierarchy h
            JOIN {edges} r ON h.entity_id = r.{inward_col}
            JOIN {entities} e ON r.{outward_col} = e.{id_col}
            WHERE {rel_membership}
                AND h.depth < {int(max_depth)}
        )
        SELECT * FROM hierarchy
        ORDER BY depth ASC
    """
    return sql, {"entity_id": entity_id, "rel_types": rel_types}
