"""DuckDB implementation of `GraphBackend` — zero-cloud-setup local storage.

Ontology semantics are resolved to concrete relationship types, and requested depths are
clamped against ontology-declared caps, BEFORE any pattern module is called: patterns never
see a semantic name and never see the ontology's depth-cap logic, only an already-concrete
`rel_types: list[str]` and a plain `int` depth.

Edge direction is canonicalized once, at load time, per each relationship type's declared
`canonical_direction` (swapping source/target columns for any type declared
`target_to_source`) — so every pattern's SQL only ever needs to walk one direction per hop.
Doing this per-query in SQL instead would mean two `UNION ALL` branches on every hop forever,
and would drag the same direction-inversion logic into the retriever facade added later.

Security note: the ported source builds its relationship-type SQL filter by joining the type
names directly into the query text with an f-string — a SQL injection surface if a type name
ever came from outside trusted config. Every relationship-type list here instead travels as a
genuine array query parameter through `SqlDialect.array_membership`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb

from graph_rag.backends.base import (
    DESCENDANT_COUNT_COLUMN,
    DESCENDANT_TYPE_COLUMN,
    order_by_declared_relationship_type,
    pivot_descendant_counts,
)
from graph_rag.dialects.duckdb import DuckDbDialect
from graph_rag.models import BlockerHit, EnrichmentResult, Entity
from graph_rag.ontology.models import Ontology
from graph_rag.ontology.resolve import effective_max_depth, resolve_semantic
from graph_rag.patterns import batch_enrichment, blocker_chains, hierarchy

# Semantic aliases this backend resolves before calling a pattern. These names are part of the
# cross-domain "meaning" vocabulary a caller-supplied ontology is expected to declare (see
# ontology/org_graph.yaml and tests/fixtures/tiny_domain.yaml) — unlike a relationship or
# entity type name, they are not domain literals: they name a role a semantic fills, not a
# fact about one specific domain's data.
_HIERARCHY_SEMANTIC = "hierarchy"
_UPSTREAM_SEMANTIC = "upstream"
_OWNERSHIP_SEMANTIC = "ownership"
_RISK_SEMANTIC = "risk"

_DEFAULT_HIERARCHY_DEPTH = 3
_DEFAULT_BLOCKER_DEPTH = 3
_ENRICH_HIERARCHY_DEPTH = 2
_ENRICH_BLOCKER_DEPTH = 2


def _parse_properties(value: Any) -> Any:
    """Parse a JSON-string `properties` column, tolerating already-structured or bad input."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    return value


def _rows_as_dicts(cursor: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    columns = [d[0] for d in cursor.description]
    rows = [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
    for row in rows:
        if "properties" in row:
            row["properties"] = _parse_properties(row["properties"])
    return rows


class DuckDBGraphBackend:
    """`GraphBackend` implementation over an in-process or on-disk DuckDB database."""

    def __init__(self, db_path: str, ontology: Ontology) -> None:
        """Connect to `db_path` (use ``":memory:"`` for an ephemeral database)."""
        self.ontology = ontology
        self.dialect = DuckDbDialect()
        self.conn = duckdb.connect(db_path)
        self._tc = ontology.table_config

    @classmethod
    def from_csv(
        cls,
        entities_csv: str | Path,
        relationships_csv: str | Path,
        ontology: Ontology,
    ) -> DuckDBGraphBackend:
        """Build an in-memory backend by loading the two CSVs into tables named per the ontology."""
        backend = cls(":memory:", ontology)
        backend._load_csv(entities_csv, relationships_csv)
        return backend

    def reload_ontology(self, ontology: Ontology) -> None:
        """Swap in a newly loaded Ontology without reconnecting — the "live swap" seam.

        The connection and its data are untouched; only which vocabulary governs the next
        query changes. Demonstrates that a table-backed ontology source's new relationship
        type takes effect on the very next call, with no restart and no change to any pattern
        or backend code — see `demo.py`'s live-swap section.
        """
        self.ontology = ontology
        self._tc = ontology.table_config

    def _load_csv(self, entities_csv: str | Path, relationships_csv: str | Path) -> None:
        tc = self._tc
        self.conn.execute(
            f"CREATE TABLE {tc.node_table} AS SELECT * FROM read_csv_auto(?, header=True)",
            [str(entities_csv)],
        )
        self.conn.execute(
            f"CREATE TABLE {tc.edge_table} AS SELECT * FROM read_csv_auto(?, header=True)",
            [str(relationships_csv)],
        )
        self._canonicalize_edge_direction()

    def _canonicalize_edge_direction(self) -> None:
        """Swap source/target for every relationship type declared `target_to_source`.

        After this runs, every relationship type's edges point in its `source_to_target`
        sense regardless of how it was declared, so pattern modules only ever need to reason
        about one direction per hop.
        """
        tc = self._tc
        to_swap = [
            rel.name for rel in self.ontology.relationship_types
            if rel.canonical_direction == "target_to_source"
        ]
        if not to_swap:
            return
        placeholders = ", ".join(f"${i + 1}" for i in range(len(to_swap)))
        self.conn.execute(
            f"""
            UPDATE {tc.edge_table}
            SET {tc.edge_source_column} = {tc.edge_target_column},
                {tc.edge_target_column} = {tc.edge_source_column}
            WHERE {tc.edge_type_column} IN ({placeholders})
            """,
            to_swap,
        )

    def _execute(self, sql: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        cursor = self.conn.execute(sql, self.dialect.bind(params))
        return _rows_as_dicts(cursor)

    def _get_entity_row(self, entity_id: str) -> dict[str, Any] | None:
        tc = self._tc
        rows = self._execute(
            f"""
            SELECT {tc.node_projection()}
            FROM {tc.node_table}
            WHERE {tc.node_id_column} = {self.dialect.param("entity_id")}
            """,
            {"entity_id": entity_id},
        )
        return rows[0] if rows else None

    # -- Pattern 1: bounded recursive hierarchy -------------------------------------------

    def get_entity_hierarchy(self, entity_id: str, direction: str = "both") -> dict:
        """Return `{"entity": dict | None, "parents": list[dict], "children": list[dict]}`."""
        result: dict[str, Any] = {"entity": None, "parents": [], "children": []}
        entity_row = self._get_entity_row(entity_id)
        if entity_row is None:
            return result
        result["entity"] = entity_row

        rel_types = resolve_semantic(self.ontology, _HIERARCHY_SEMANTIC)
        depth = effective_max_depth(self.ontology, rel_types, _DEFAULT_HIERARCHY_DEPTH)

        if direction in ("up", "both"):
            sql, params = hierarchy.render(
                self.dialect, self.ontology, rel_types, entity_id, "up", depth
            )
            result["parents"] = self._execute(sql, params)
        if direction in ("down", "both"):
            sql, params = hierarchy.render(
                self.dialect, self.ontology, rel_types, entity_id, "down", depth
            )
            result["children"] = self._execute(sql, params)
        return result

    # -- Pattern 2: blocker chains ---------------------------------------------------------

    def find_blockers(self, entity_id: str, max_depth: int = 3) -> list[BlockerHit]:
        """Find entities that transitively block/depend-on-block `entity_id`."""
        rel_types = resolve_semantic(self.ontology, _UPSTREAM_SEMANTIC)
        depth = effective_max_depth(self.ontology, rel_types, max_depth)
        sql, params = blocker_chains.render(
            self.dialect, self.ontology, rel_types, entity_id, depth
        )
        rows = self._execute(sql, params)
        return [BlockerHit(**row) for row in rows]

    # -- Pattern 3: batch enrichment ---------------------------------------------------------

    def enrich_entities_batch(self, entity_ids: list[str]) -> dict[str, EnrichmentResult]:
        """Enrich every id in `entity_ids` (hierarchy, blockers, risks, owners) in one query."""
        if not entity_ids:
            return {}

        hierarchy_rel_types = resolve_semantic(self.ontology, _HIERARCHY_SEMANTIC)
        blocker_rel_types = resolve_semantic(self.ontology, _UPSTREAM_SEMANTIC)
        risk_rel_types = resolve_semantic(self.ontology, _RISK_SEMANTIC)
        ownership_rel_types = resolve_semantic(self.ontology, _OWNERSHIP_SEMANTIC)

        hierarchy_depth = effective_max_depth(
            self.ontology, hierarchy_rel_types, _ENRICH_HIERARCHY_DEPTH
        )
        blocker_depth = effective_max_depth(
            self.ontology, blocker_rel_types, _ENRICH_BLOCKER_DEPTH
        )

        sql, params = batch_enrichment.render(
            self.dialect,
            self.ontology,
            entity_ids,
            hierarchy_rel_types,
            blocker_rel_types,
            risk_rel_types,
            ownership_rel_types,
            hierarchy_depth,
            blocker_depth,
        )
        rows = self._execute(sql, params)

        enriched: dict[str, EnrichmentResult] = {}
        for row in rows:
            enriched[row["entity_id"]] = EnrichmentResult(
                entity_id=row["entity_id"],
                hierarchy=[Entity(**h) for h in row["hierarchy"]],
                blockers=[BlockerHit(**b) for b in row["blockers"]],
                risks=[Entity(**r) for r in row["risks"]],
                owners=[Entity(**o) for o in row["owners"]],
            )
        return enriched

    # -- Non-pattern methods: generic traversal, risks, owners, lookup, summary ------------
    #
    # These are backend-specific SQL builders rather than shared `patterns/` modules (per the
    # milestone brief, this split is a judgment call): each is small enough, and different
    # enough between engines in its ORDER BY / LIMIT shape, that a later BigQuery backend is
    # expected to re-implement them directly rather than share a pattern module.

    def find_risks_for_entity(self, entity_id: str) -> list[Entity]:
        """Find risks that threaten `entity_id` or any of its descendants.

        Relies on the ontology's own domain/range guarantee for the `threats` semantic's
        relationship types (only a `Risk`-typed entity can be the source of a `threatens`
        edge) rather than an extra `type = 'Risk'` literal filter.
        """
        tc = self._tc
        hierarchy_rel_types = resolve_semantic(self.ontology, _HIERARCHY_SEMANTIC)
        risk_rel_types = resolve_semantic(self.ontology, _RISK_SEMANTIC)
        depth = effective_max_depth(self.ontology, hierarchy_rel_types, _DEFAULT_HIERARCHY_DEPTH)

        entity_param = self.dialect.param("entity_id")
        hierarchy_membership = self.dialect.array_membership(
            f"r.{tc.edge_type_column}", "hierarchy_rel_types"
        )
        risk_membership = self.dialect.array_membership(
            f"r.{tc.edge_type_column}", "risk_rel_types"
        )

        sql = f"""
            WITH RECURSIVE children AS (
                SELECT e.{tc.node_id_column} AS entity_id, 1 AS depth
                FROM {tc.edge_table} r
                JOIN {tc.node_table} e ON r.{tc.edge_source_column} = e.{tc.node_id_column}
                WHERE r.{tc.edge_target_column} = {entity_param}
                    AND {hierarchy_membership}

                UNION ALL

                SELECT e.{tc.node_id_column}, c.depth + 1
                FROM children c
                JOIN {tc.edge_table} r ON c.entity_id = r.{tc.edge_target_column}
                JOIN {tc.node_table} e ON r.{tc.edge_source_column} = e.{tc.node_id_column}
                WHERE {hierarchy_membership}
                    AND c.depth < {int(depth)}
            ),
            entity_and_children AS (
                SELECT {entity_param} AS entity_id
                UNION
                SELECT entity_id FROM children
            )
            SELECT {tc.node_projection("risk")}
            FROM {tc.edge_table} r
            JOIN {tc.node_table} risk ON r.{tc.edge_source_column} = risk.{tc.node_id_column}
            JOIN entity_and_children target ON r.{tc.edge_target_column} = target.entity_id
            WHERE {risk_membership}
        """
        rows = self._execute(
            sql,
            {
                "entity_id": entity_id,
                "hierarchy_rel_types": hierarchy_rel_types,
                "risk_rel_types": risk_rel_types,
            },
        )
        return [Entity(**row) for row in rows]

    def get_entity_owners(self, entity_id: str) -> list[Entity]:
        """Find people who own or are accountable for `entity_id`, owners first.

        "Owners first" is the ontology's ordering, not this method's: `resolve_semantic` returns
        the `ownership` semantic's relationship types in declared order, and
        `order_by_declared_relationship_type` sorts by position in that list. A domain that
        declares its ownership types the other way round gets the other order, with no code
        change here.

        Relies on the ontology's domain/range guarantee for `owns`/`accountable_for` (only a
        `Person`-typed entity can be their source) rather than an extra type literal filter.
        """
        tc = self._tc
        ownership_rel_types = resolve_semantic(self.ontology, _OWNERSHIP_SEMANTIC)
        entity_param = self.dialect.param("entity_id")
        membership = self.dialect.array_membership(f"r.{tc.edge_type_column}", "rel_types")

        sql = f"""
            SELECT DISTINCT
                {tc.node_projection("p")},
                r.{tc.edge_type_column} AS relationship_type
            FROM {tc.edge_table} r
            JOIN {tc.node_table} p ON r.{tc.edge_source_column} = p.{tc.node_id_column}
            WHERE r.{tc.edge_target_column} = {entity_param}
                AND {membership}
        """
        rows = self._execute(sql, {"entity_id": entity_id, "rel_types": ownership_rel_types})
        rows = order_by_declared_relationship_type(rows, ownership_rel_types)
        return [Entity(**{k: v for k, v in row.items() if k != "relationship_type"}) for row in rows]

    def traverse_relationships(
        self,
        start_id: str,
        rel_type: str | list[str],
        depth: int = 1,
        direction: str = "out",
    ) -> list[dict]:
        """Traverse relationship(s) from `start_id`. Raises ValueError for multi-hop "both"."""
        tc = self._tc
        rel_types = [rel_type] if isinstance(rel_type, str) else rel_type
        start_param = self.dialect.param("start_id")
        membership = self.dialect.array_membership(f"r.{tc.edge_type_column}", "rel_types")

        if depth == 1:
            if direction == "out":
                join_col, filter_col = tc.edge_target_column, tc.edge_source_column
            elif direction == "in":
                join_col, filter_col = tc.edge_source_column, tc.edge_target_column
            else:
                raise ValueError("direction must be 'out' or 'in' for this backend's 'both' path")
            sql = f"""
                SELECT DISTINCT
                    e.{tc.node_id_column} AS entity_id, e.{tc.node_name_column} AS name,
                    e.{tc.node_type_column} AS type, e.{tc.node_status_column} AS status,
                    r.{tc.edge_type_column} AS relationship_type, 1 AS depth
                FROM {tc.edge_table} r
                JOIN {tc.node_table} e ON r.{join_col} = e.{tc.node_id_column}
                WHERE r.{filter_col} = {start_param}
                    AND {membership}
                LIMIT 100
            """
            return self._execute(sql, {"start_id": start_id, "rel_types": rel_types})

        if direction not in ("out", "in"):
            raise ValueError("Multi-hop 'both' direction not supported. Use 'in' or 'out'.")

        outward_col = tc.edge_target_column if direction == "out" else tc.edge_source_column
        inward_col = tc.edge_source_column if direction == "out" else tc.edge_target_column
        depth_param = self.dialect.param("depth")

        sql = f"""
            WITH RECURSIVE traversal AS (
                SELECT
                    e.{tc.node_id_column} AS entity_id, e.{tc.node_name_column} AS name,
                    e.{tc.node_type_column} AS type, e.{tc.node_status_column} AS status,
                    r.{tc.edge_type_column} AS relationship_type, 1 AS depth
                FROM {tc.edge_table} r
                JOIN {tc.node_table} e ON r.{outward_col} = e.{tc.node_id_column}
                WHERE r.{inward_col} = {start_param}
                    AND {membership}

                UNION ALL

                SELECT
                    e.{tc.node_id_column}, e.{tc.node_name_column},
                    e.{tc.node_type_column}, e.{tc.node_status_column},
                    r.{tc.edge_type_column}, t.depth + 1
                FROM traversal t
                JOIN {tc.edge_table} r ON t.entity_id = r.{inward_col}
                JOIN {tc.node_table} e ON r.{outward_col} = e.{tc.node_id_column}
                WHERE {membership}
                    AND t.depth < {depth_param}
            )
            SELECT * FROM traversal
            ORDER BY depth ASC
            LIMIT 100
        """
        return self._execute(sql, {"start_id": start_id, "rel_types": rel_types, "depth": depth})

    def find_by_name(
        self, name: str, entity_type: str | None = None, exact: bool = False
    ) -> Entity | None:
        """Find a single entity by name, exact or partial (case-insensitive) match."""
        tc = self._tc
        name_param = self.dialect.param("name")
        select = f"SELECT {tc.node_projection()} FROM {tc.node_table}"
        if exact:
            where = f"WHERE LOWER({tc.node_name_column}) = LOWER({name_param})"
        else:
            where = f"WHERE LOWER({tc.node_name_column}) LIKE LOWER('%' || {name_param} || '%')"

        params: dict[str, Any] = {"name": name}
        if entity_type:
            where += f" AND {tc.node_type_column} = {self.dialect.param('entity_type')}"
            params["entity_type"] = entity_type

        rows = self._execute(f"{select} {where} LIMIT 1", params)
        return Entity(**rows[0]) if rows else None

    def get_descendant_counts(
        self, root_type: str, count_types: list[str] | None = None
    ) -> list[dict]:
        """Per-root columns plus RECURSIVE (transitive) descendant counts by entity type.

        Two deliberate departures from the ported source, whose equivalent counted only DIRECT
        children of a hardcoded root type into hardcoded per-type columns:

        * The walk is recursive. On this repo's strict tree (task -> project -> initiative ->
          goal), counting direct children alone reports zero for everything two or more hops
          down.
        * The types are arguments, not literals. This is a method on the `GraphBackend`
          Protocol — the advertised generic API — so baking one domain's type names into it
          contradicted the registry's whole premise. `root_type` travels as a query parameter,
          and the per-type breakdown is pivoted in Python by `pivot_descendant_counts` rather
          than as one `CASE WHEN type = '...'` literal per counted type.
        """
        tc = self._tc
        count_types = count_types if count_types is not None else self.ontology.entity_type_names()
        hierarchy_rel_types = resolve_semantic(self.ontology, _HIERARCHY_SEMANTIC)
        depth = effective_max_depth(self.ontology, hierarchy_rel_types, _DEFAULT_HIERARCHY_DEPTH)
        membership = self.dialect.array_membership(f"r.{tc.edge_type_column}", "rel_types")
        root_param = self.dialect.param("root_type")

        sql = f"""
            WITH RECURSIVE descendants AS (
                SELECT
                    g.{tc.node_id_column} AS root_id,
                    e.{tc.node_id_column} AS entity_id, e.{tc.node_type_column} AS type,
                    1 AS depth
                FROM {tc.node_table} g
                JOIN {tc.edge_table} r ON r.{tc.edge_target_column} = g.{tc.node_id_column}
                JOIN {tc.node_table} e ON r.{tc.edge_source_column} = e.{tc.node_id_column}
                WHERE {membership}

                UNION ALL

                SELECT d.root_id, e.{tc.node_id_column}, e.{tc.node_type_column}, d.depth + 1
                FROM descendants d
                JOIN {tc.edge_table} r ON r.{tc.edge_target_column} = d.entity_id
                JOIN {tc.node_table} e ON r.{tc.edge_source_column} = e.{tc.node_id_column}
                WHERE {membership}
                    AND d.depth < {int(depth)}
            ),
            counts AS (
                SELECT root_id, type AS {DESCENDANT_TYPE_COLUMN},
                       COUNT(DISTINCT entity_id) AS {DESCENDANT_COUNT_COLUMN}
                FROM descendants
                GROUP BY root_id, type
            )
            SELECT {tc.node_projection("g")},
                   c.{DESCENDANT_TYPE_COLUMN}, c.{DESCENDANT_COUNT_COLUMN}
            FROM {tc.node_table} g
            LEFT JOIN counts c ON g.{tc.node_id_column} = c.root_id
            WHERE g.{tc.node_type_column} = {root_param}
            ORDER BY name
        """
        rows = self._execute(sql, {"rel_types": hierarchy_rel_types, "root_type": root_type})
        return pivot_descendant_counts(rows, count_types)
