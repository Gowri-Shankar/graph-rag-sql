"""Pattern 3: batch enrichment via UNNEST — killing the N+1 problem.

After a vector search returns the top-k entities for a RAG answer, each one typically needs
its graph context (ancestors, blockers, risks, owners) — naively, that's 4 queries per entity,
4×k round trips. This pattern seeds every recursive CTE and join with `IN UNNEST(@entity_ids)`
(rendered via `array_membership`) instead of a single `entity_id`, so ALL k entities are
enriched in exactly one query, with results grouped back per entity via
`LEFT JOIN ... ON entity_id = <aggregate>.<key>` and `COALESCE(..., <empty array>)` so a
caller never has to special-case "no blockers" vs "no blockers column".

Complexity: same recursive-CTE cost per hop as Patterns 1 and 2, paid once for the whole
batch rather than once per entity — the entire point of the pattern.

No domain literals: every relationship-type list (`hierarchy_rel_types`, `blocker_rel_types`,
`risk_rel_types`, `ownership_rel_types`) is pre-resolved by the caller; table/column names
come from `ontology.table_config`. Deviation from the ported source: rather than filtering by
an entity-type literal (e.g. requiring the source of a "threat" edge to be a specific
entity type), this pattern relies on the ontology's own domain/range guarantee — each
relationship type already restricts which entity type may originate it (enforced by
`validate_edges` at data-write time) — so the relationship-type filter alone is sufficient and
no entity-type literal is needed here at all.
"""

from __future__ import annotations

from graph_rag.dialects.base import SqlDialect
from graph_rag.ontology.models import Ontology


def render(
    dialect: SqlDialect,
    ontology: Ontology,
    entity_ids: list[str],
    hierarchy_rel_types: list[str],
    blocker_rel_types: list[str],
    risk_rel_types: list[str],
    ownership_rel_types: list[str],
    hierarchy_max_depth: int,
    blocker_max_depth: int,
) -> tuple[str, dict[str, object]]:
    """Render the batch enrichment query.

    Args:
        dialect: The SQL dialect to render against.
        ontology: Supplies table/column names via `ontology.table_config`.
        entity_ids: The batch of entities to enrich.
        hierarchy_rel_types: Resolved relationship types for the ancestor walk.
        blocker_rel_types: Resolved relationship types for the blocker-chain walk.
        risk_rel_types: Resolved relationship types for the threat lookup.
        ownership_rel_types: Resolved relationship types for the ownership lookup.
        hierarchy_max_depth: Depth bound for the ancestor walk (already clamped).
        blocker_max_depth: Depth bound for the blocker-chain walk (already clamped).

    Returns:
        `(sql, params)`. Each result row has `entity_id` plus `hierarchy`, `blockers`,
        `risks`, `owners` — arrays of structs, empty (never null) when a group has no members.
    """
    tc = ontology.table_config
    entities = dialect.qualify_table(tc.node_table)
    edges = dialect.qualify_table(tc.edge_table)
    id_col, name_col, type_col, status_col = (
        tc.node_id_column,
        tc.node_name_column,
        tc.node_type_column,
        tc.node_status_column,
    )
    src_col, dst_col, edge_type_col = (
        tc.edge_source_column,
        tc.edge_target_column,
        tc.edge_type_column,
    )

    entity_ids_membership_src = dialect.array_membership(f"r.{src_col}", "entity_ids")
    entity_ids_membership_dst = dialect.array_membership(f"r.{dst_col}", "entity_ids")
    entity_ids_membership_id = dialect.array_membership(id_col, "entity_ids")

    hierarchy_membership = dialect.array_membership(f"r.{edge_type_col}", "hierarchy_rel_types")
    blocker_membership = dialect.array_membership(f"r.{edge_type_col}", "blocker_rel_types")
    risk_membership = dialect.array_membership(f"r.{edge_type_col}", "risk_rel_types")
    ownership_membership = dialect.array_membership(f"r.{edge_type_col}", "ownership_rel_types")

    rel_chain_seed = dialect.array_literal([f"r.{edge_type_col}"])
    rel_chain_grow = dialect.array_append("bc.rel_chain", dialect.array_literal([f"r.{edge_type_col}"]))
    name_chain_seed = dialect.array_literal([f"e.{name_col}"])
    name_chain_grow = dialect.array_append("bc.name_chain", dialect.array_literal([f"e.{name_col}"]))

    entity_fields = [
        ("entity_id", "string"),
        ("name", "string"),
        ("type", "string"),
        ("status", "string"),
    ]
    blocker_fields = [
        ("entity_id", "string"),
        ("name", "string"),
        ("status", "string"),
        ("distance", "int"),
        ("rel_chain", "string[]"),
        ("name_chain", "string[]"),
    ]

    parents_agg = dialect.struct_agg("parents", ["entity_id", "name", "type", "status"])
    blockers_agg = dialect.struct_agg(
        "blockers", ["entity_id", "name", "status", "distance", "rel_chain", "name_chain"]
    )
    risks_agg = dialect.struct_agg("risks", ["entity_id", "name", "type", "status"])
    owners_agg = dialect.struct_agg("owners", ["entity_id", "name", "type", "status"])

    sql = f"""
        WITH RECURSIVE
        parent_hierarchy AS (
            SELECT
                r.{src_col} AS child_id,
                e.{id_col} AS entity_id, e.{name_col} AS name,
                e.{type_col} AS type, e.{status_col} AS status,
                1 AS depth
            FROM {edges} r
            JOIN {entities} e ON r.{dst_col} = e.{id_col}
            WHERE {entity_ids_membership_src}
                AND {hierarchy_membership}

            UNION ALL

            SELECT
                ph.child_id,
                e.{id_col}, e.{name_col}, e.{type_col}, e.{status_col},
                ph.depth + 1
            FROM parent_hierarchy ph
            JOIN {edges} r ON ph.entity_id = r.{src_col}
            JOIN {entities} e ON r.{dst_col} = e.{id_col}
            WHERE {hierarchy_membership}
                AND ph.depth < {int(hierarchy_max_depth)}
        ),
        blocker_chain AS (
            SELECT
                r.{dst_col} AS blocked_id,
                e.{id_col} AS entity_id, e.{name_col} AS name, e.{status_col} AS status,
                1 AS distance,
                {rel_chain_seed} AS rel_chain,
                {name_chain_seed} AS name_chain
            FROM {edges} r
            JOIN {entities} e ON r.{src_col} = e.{id_col}
            WHERE {entity_ids_membership_dst}
                AND {blocker_membership}

            UNION ALL

            SELECT
                bc.blocked_id,
                e.{id_col}, e.{name_col}, e.{status_col},
                bc.distance + 1,
                {rel_chain_grow},
                {name_chain_grow}
            FROM blocker_chain bc
            JOIN {edges} r ON bc.entity_id = r.{dst_col}
            JOIN {entities} e ON r.{src_col} = e.{id_col}
            WHERE {blocker_membership}
                AND bc.distance < {int(blocker_max_depth)}
        ),
        target_entities AS (
            SELECT {id_col} AS entity_id
            FROM {entities}
            WHERE {entity_ids_membership_id}
        ),
        parents AS (
            SELECT child_id, {parents_agg}
            FROM parent_hierarchy
            GROUP BY child_id
        ),
        blockers AS (
            SELECT blocked_id, {blockers_agg}
            FROM blocker_chain
            GROUP BY blocked_id
        ),
        risks AS (
            SELECT r.{dst_col} AS threatened_id, {risks_agg}
            FROM {edges} r
            JOIN {entities} e ON r.{src_col} = e.{id_col}
            WHERE {entity_ids_membership_dst}
                AND {risk_membership}
            GROUP BY r.{dst_col}
        ),
        owners AS (
            SELECT r.{dst_col} AS owned_id, {owners_agg}
            FROM {edges} r
            JOIN {entities} e ON r.{src_col} = e.{id_col}
            WHERE {entity_ids_membership_dst}
                AND {ownership_membership}
            GROUP BY r.{dst_col}
        )
        SELECT
            te.entity_id,
            COALESCE(p.parents, {dialect.empty_array(entity_fields)}) AS hierarchy,
            COALESCE(b.blockers, {dialect.empty_array(blocker_fields)}) AS blockers,
            COALESCE(r.risks, {dialect.empty_array(entity_fields)}) AS risks,
            COALESCE(o.owners, {dialect.empty_array(entity_fields)}) AS owners
        FROM target_entities te
        LEFT JOIN parents p ON te.entity_id = p.child_id
        LEFT JOIN blockers b ON te.entity_id = b.blocked_id
        LEFT JOIN risks r ON te.entity_id = r.threatened_id
        LEFT JOIN owners o ON te.entity_id = o.owned_id
    """
    params = {
        "entity_ids": entity_ids,
        "hierarchy_rel_types": hierarchy_rel_types,
        "blocker_rel_types": blocker_rel_types,
        "risk_rel_types": risk_rel_types,
        "ownership_rel_types": ownership_rel_types,
    }
    return sql, params
