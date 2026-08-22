"""Pattern 2: blocker-chain detection, collecting the traversal path in arrays — the showcase.

Follows a transitive "impedes" relationship backward from a target entity, the same way
Pattern 1 walks a hierarchy — but instead of returning only the endpoints, each row carries
the full path traveled to reach the target, as parallel `rel_chain` (relationship types) and
`name_chain` (entity names) arrays. The answer isn't "these 4 things impede Atlas", it's *how*
each one impedes it.

Complexity/limits: same shape as Pattern 1 — one recursive self-join per hop, bounded by
`max_depth`. Array columns are only ever compared, never used in `SELECT DISTINCT` (that's
invalid on BigQuery); deduplication uses `top_n_per_group` (rendered as `QUALIFY` on both
DuckDB and BigQuery) over the scalar `(entity_id, distance)` pair instead.

No domain literals: `rel_types` is a resolved list (typically `resolve_semantic(ontology,
"upstream")`); table/column names come from `ontology.table_config`. The final ordering by a
status `CASE` (`blocked` / `at_risk` / `delayed` first) is retained from the ported source —
status values are a display concern, not part of the ontology's entity/relationship vocabulary.
"""

from __future__ import annotations

from graph_rag.dialects.base import SqlDialect
from graph_rag.ontology.models import Ontology


def render(
    dialect: SqlDialect,
    ontology: Ontology,
    rel_types: list[str],
    entity_id: str,
    max_depth: int,
) -> tuple[str, dict[str, object]]:
    """Render the recursive blocker-chain query.

    Args:
        dialect: The SQL dialect to render against.
        ontology: Supplies table/column names via `ontology.table_config`.
        rel_types: Concrete, pre-resolved relationship type names to traverse.
        entity_id: The entity to find blockers/dependencies for.
        max_depth: Depth bound, already clamped by `effective_max_depth`.

    Returns:
        `(sql, params)`. `params` has keys `"entity_id"`, `"max_depth"`, and `"rel_types"`.
        Each result row carries `entity_id`, `name`, `status`, `distance`, `rel_chain`
        (grows outward from the anchor — reverse it to read farthest-blocker-first), and
        `name_chain` (same order as `rel_chain`, plus the blocker's own name at each hop).
    """
    tc = ontology.table_config
    entities = dialect.qualify_table(tc.node_table)
    edges = dialect.qualify_table(tc.edge_table)
    id_col, name_col, status_col = tc.node_id_column, tc.node_name_column, tc.node_status_column
    src_col, dst_col, edge_type_col = (
        tc.edge_source_column,
        tc.edge_target_column,
        tc.edge_type_column,
    )

    entity_param = dialect.param("entity_id")
    max_depth_param = dialect.param("max_depth")
    rel_membership = dialect.array_membership(f"r.{edge_type_col}", "rel_types")

    rel_chain_seed = dialect.array_literal([f"r.{edge_type_col}"])
    name_chain_seed = dialect.array_literal([f"e.{name_col}"])
    rel_chain_grow = dialect.array_append("bc.rel_chain", dialect.array_literal([f"r.{edge_type_col}"]))
    name_chain_grow = dialect.array_append(
        "bc.name_chain", dialect.array_literal([f"e.{name_col}"])
    )
    dedup_clause = dialect.top_n_per_group(["entity_id", "distance"], "name", 1)

    sql = f"""
        WITH RECURSIVE blocker_chain AS (
            SELECT
                e.{id_col} AS entity_id, e.{name_col} AS name, e.{status_col} AS status,
                1 AS distance,
                {rel_chain_seed} AS rel_chain,
                {name_chain_seed} AS name_chain
            FROM {edges} r
            JOIN {entities} e ON r.{src_col} = e.{id_col}
            WHERE r.{dst_col} = {entity_param}
                AND {rel_membership}

            UNION ALL

            SELECT
                e.{id_col} AS entity_id, e.{name_col} AS name, e.{status_col} AS status,
                bc.distance + 1,
                {rel_chain_grow},
                {name_chain_grow}
            FROM blocker_chain bc
            JOIN {edges} r ON bc.entity_id = r.{dst_col}
            JOIN {entities} e ON r.{src_col} = e.{id_col}
            WHERE {rel_membership}
                AND bc.distance < {max_depth_param}
        )
        SELECT * FROM blocker_chain
        {dedup_clause}
        ORDER BY distance ASC,
            CASE status
                WHEN 'blocked' THEN 1
                WHEN 'at_risk' THEN 2
                WHEN 'delayed' THEN 3
                ELSE 4
            END
        LIMIT 50
    """
    return sql, {"entity_id": entity_id, "max_depth": max_depth, "rel_types": rel_types}
