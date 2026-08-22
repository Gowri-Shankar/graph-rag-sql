"""BigQuery implementation of `GraphBackend` — the production path.

Pattern 1 (hierarchy), pattern 2 (blocker chains), and pattern 3 (batch enrichment) call the
exact same `graph_rag.patterns` render functions `DuckDBGraphBackend` calls — only the
`SqlDialect` passed in differs. The remaining methods (risks, owners, generic traversal, name
lookup, goals summary) are backend-specific SQL builders, same as in `DuckDBGraphBackend`: per
the milestone design, each is small and different enough between engines in its ORDER BY/LIMIT
shape that sharing a pattern module isn't worth it.

Security note: exactly like the DuckDB port, the ported source's `traverse_relationships`
builds its relationship-type SQL filter by joining type names directly into the query text —
a SQL injection surface if a type name ever came from outside trusted config. Every
relationship-type list here instead travels as a genuine `bigquery.ArrayQueryParameter` through
`SqlDialect.array_membership` — never an f-string.

Import note: `google.cloud.bigquery` is imported lazily, inside `__init__`, so importing
`graph_rag` (or even this module) never requires the `[bigquery]` extra to be installed.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from graph_rag.dialects.bigquery import BigQueryDialect
from graph_rag.models import BlockerHit, EnrichmentResult, Entity
from graph_rag.ontology.models import Ontology
from graph_rag.ontology.resolve import effective_max_depth, resolve_semantic
from graph_rag.patterns import batch_enrichment, blocker_chains, hierarchy

if TYPE_CHECKING:
    from google.cloud import bigquery

_HIERARCHY_SEMANTIC = "hierarchy"
_UPSTREAM_SEMANTIC = "upstream"
_OWNERSHIP_SEMANTIC = "ownership"
_RISK_SEMANTIC = "risk"

_DEFAULT_HIERARCHY_DEPTH = 3
_DEFAULT_BLOCKER_DEPTH = 3
_ENRICH_HIERARCHY_DEPTH = 2
_ENRICH_BLOCKER_DEPTH = 2


def _parse_properties(value: Any) -> Any:
    """Parse a JSON-string `properties` field, tolerating already-structured or bad input."""
    if isinstance(value, str):
        import json

        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return value
    return value


def _struct_rows(items: Any) -> list[dict[str, Any]]:
    """Convert a BigQuery ARRAY<STRUCT> result field into plain dicts, parsing `properties`."""
    rows = []
    for item in items or []:
        row = dict(item.items()) if hasattr(item, "items") else dict(item)
        if "properties" in row:
            row["properties"] = _parse_properties(row["properties"])
        rows.append(row)
    return rows


class BigQueryGraphBackend:
    """`GraphBackend` implementation over Google BigQuery."""

    def __init__(
        self,
        ontology: Ontology,
        project_id: str | None = None,
        dataset_id: str | None = None,
        use_query_cache: bool = True,
    ) -> None:
        """Connect to BigQuery.

        Args:
            ontology: The vocabulary this backend resolves semantics and depth caps against.
            project_id: GCP project ID. Defaults to the `GCP_PROJECT_ID` env var.
            dataset_id: BigQuery dataset ID. Defaults to the `BQ_DATASET_ID` env var.
            use_query_cache: Whether BigQuery may serve a query from its results cache. A cache
                hit reports `total_bytes_processed = 0` (and isn't billed), which understates
                real traversal cost — `scripts/benchmark.py` passes `False` so its bytes-billed
                and cost figures reflect a real scan, not a cached rerun.

        Raises:
            ImportError: If the `google-cloud-bigquery` package isn't installed.
            ValueError: If neither a project_id/dataset_id argument nor the corresponding env
                var is set — this backend never falls back to a hardcoded default project.
        """
        try:
            from google.cloud import bigquery
        except ImportError as exc:
            raise ImportError(
                "BigQueryGraphBackend requires the 'bigquery' extra. Install it with: "
                'pip install "graph-rag-sql[bigquery]"'
            ) from exc

        project_id = project_id or os.environ.get("GCP_PROJECT_ID")
        dataset_id = dataset_id or os.environ.get("BQ_DATASET_ID")
        if not project_id or not dataset_id:
            raise ValueError(
                "BigQueryGraphBackend requires a project_id and dataset_id, supplied as "
                "constructor arguments or via the GCP_PROJECT_ID / BQ_DATASET_ID env vars."
            )

        self._bigquery = bigquery
        self.ontology = ontology
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.use_query_cache = use_query_cache
        self.dialect = BigQueryDialect(project_id, dataset_id)
        self.client = bigquery.Client(project=project_id)
        self._tc = ontology.table_config
        # The most recently finished QueryJob, so a caller (e.g. scripts/benchmark.py) can read
        # total_bytes_processed for cost reporting without this backend taking on a
        # billing-reporting responsibility of its own.
        self.last_query_job: Any = None

    def reload_ontology(self, ontology: Ontology) -> None:
        """Swap in a newly loaded Ontology without reconnecting — same seam as the DuckDB backend."""
        self.ontology = ontology
        self._tc = ontology.table_config

    # -- Query execution ---------------------------------------------------------------------

    def _query_parameters(self, params: dict[str, Any]) -> list[bigquery.ScalarQueryParameter]:
        bigquery = self._bigquery
        query_params = []
        for key, value in params.items():
            if isinstance(value, list):
                query_params.append(bigquery.ArrayQueryParameter(key, "STRING", value))
            elif isinstance(value, bool):
                query_params.append(bigquery.ScalarQueryParameter(key, "BOOL", value))
            elif isinstance(value, int):
                query_params.append(bigquery.ScalarQueryParameter(key, "INT64", value))
            else:
                query_params.append(bigquery.ScalarQueryParameter(key, "STRING", str(value)))
        return query_params

    def _execute(self, sql: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        job_config = self._bigquery.QueryJobConfig(
            query_parameters=self._query_parameters(params),
            use_query_cache=self.use_query_cache,
        )
        query_job = self.client.query(sql, job_config=job_config)
        self.last_query_job = query_job
        rows = []
        for row in query_job.result():
            d = dict(row.items())
            if "properties" in d:
                d["properties"] = _parse_properties(d["properties"])
            rows.append(d)
        return rows

    def _get_entity_row(self, entity_id: str) -> dict[str, Any] | None:
        tc = self._tc
        entities = self.dialect.qualify_table(tc.node_table)
        rows = self._execute(
            f"""
            SELECT {tc.node_id_column} AS entity_id, {tc.node_name_column} AS name,
                   {tc.node_type_column} AS type, {tc.node_status_column} AS status,
                   owner_id, description, priority, risk_level, properties,
                   created_at, updated_at
            FROM {entities}
            WHERE {tc.node_id_column} = {self.dialect.param("entity_id")}
            """,
            {"entity_id": entity_id},
        )
        return rows[0] if rows else None

    # -- Pattern 1: bounded recursive hierarchy -----------------------------------------------

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

    # -- Pattern 2: blocker chains --------------------------------------------------------------

    def find_blockers(self, entity_id: str, max_depth: int = 3) -> list[BlockerHit]:
        """Find entities that transitively block/depend-on-block `entity_id`."""
        rel_types = resolve_semantic(self.ontology, _UPSTREAM_SEMANTIC)
        depth = effective_max_depth(self.ontology, rel_types, max_depth)
        sql, params = blocker_chains.render(
            self.dialect, self.ontology, rel_types, entity_id, depth
        )
        rows = self._execute(sql, params)
        return [BlockerHit(**row) for row in rows]

    # -- Pattern 3: batch enrichment --------------------------------------------------------------

    def enrich_entities_batch(self, entity_ids: list[str]) -> dict[str, EnrichmentResult]:
        """Enrich every id in `entity_ids` in one query.

        Matches the ported source's behavior: unlike every other method here, this one catches
        exceptions and degrades to `{}` instead of raising. Batch enrichment feeds RAG context
        gathering, where a partial or missing answer beats an unhandled exception aborting the
        whole request; every other method here is a direct, single-purpose lookup where the
        caller is expected to handle (or propagate) a real failure.
        """
        if not entity_ids:
            return {}

        hierarchy_rel_types = resolve_semantic(self.ontology, _HIERARCHY_SEMANTIC)
        blocker_rel_types = resolve_semantic(self.ontology, _UPSTREAM_SEMANTIC)
        risk_rel_types = resolve_semantic(self.ontology, _RISK_SEMANTIC)
        ownership_rel_types = resolve_semantic(self.ontology, _OWNERSHIP_SEMANTIC)

        hierarchy_depth = effective_max_depth(
            self.ontology, hierarchy_rel_types, _ENRICH_HIERARCHY_DEPTH
        )
        blocker_depth = effective_max_depth(self.ontology, blocker_rel_types, _ENRICH_BLOCKER_DEPTH)

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

        try:
            rows = self._execute(sql, params)
            enriched: dict[str, EnrichmentResult] = {}
            for row in rows:
                enriched[row["entity_id"]] = EnrichmentResult(
                    entity_id=row["entity_id"],
                    hierarchy=[Entity(**h) for h in _struct_rows(row["hierarchy"])],
                    blockers=[BlockerHit(**b) for b in _struct_rows(row["blockers"])],
                    risks=[Entity(**r) for r in _struct_rows(row["risks"])],
                    owners=[Entity(**o) for o in _struct_rows(row["owners"])],
                )
            return enriched
        except Exception:  # noqa: BLE001 — deliberate: see docstring, ported from the source
            return {}

    # -- Non-pattern methods: generic traversal, risks, owners, lookup, summary --------------

    def find_risks_for_entity(self, entity_id: str) -> list[Entity]:
        """Find risks that threaten `entity_id` or any of its descendants, most severe first.

        Relies on the ontology's own domain/range guarantee for the risk semantic's
        relationship types (only a `Risk`-typed entity can source a `threatens` edge) rather
        than an extra `type = 'Risk'` literal filter — same deviation as the DuckDB backend.
        Retains the source's status-priority ordering (materialized first) since that ORDER BY
        is a display concern, not a domain literal in the "no literals in patterns/" sense.
        """
        tc = self._tc
        entities = self.dialect.qualify_table(tc.node_table)
        edges = self.dialect.qualify_table(tc.edge_table)
        hierarchy_rel_types = resolve_semantic(self.ontology, _HIERARCHY_SEMANTIC)
        risk_rel_types = resolve_semantic(self.ontology, _RISK_SEMANTIC)
        depth = effective_max_depth(self.ontology, hierarchy_rel_types, _DEFAULT_HIERARCHY_DEPTH)

        entity_param = self.dialect.param("entity_id")
        hierarchy_membership = self.dialect.array_membership(
            f"r.{tc.edge_type_column}", "hierarchy_rel_types"
        )
        risk_membership = self.dialect.array_membership(f"r.{tc.edge_type_column}", "risk_rel_types")

        sql = f"""
            WITH RECURSIVE children AS (
                SELECT e.{tc.node_id_column} AS entity_id, 1 AS depth
                FROM {edges} r
                JOIN {entities} e ON r.{tc.edge_source_column} = e.{tc.node_id_column}
                WHERE r.{tc.edge_target_column} = {entity_param}
                    AND {hierarchy_membership}

                UNION ALL

                SELECT e.{tc.node_id_column}, c.depth + 1
                FROM children c
                JOIN {edges} r ON c.entity_id = r.{tc.edge_source_column}
                JOIN {entities} e ON r.{tc.edge_target_column} = e.{tc.node_id_column}
                WHERE {hierarchy_membership}
                    AND c.depth < {int(depth)}
            ),
            entity_and_children AS (
                SELECT {entity_param} AS entity_id
                UNION DISTINCT
                SELECT entity_id FROM children
            )
            SELECT
                risk.{tc.node_id_column} AS entity_id, risk.{tc.node_name_column} AS name,
                risk.{tc.node_type_column} AS type, risk.{tc.node_status_column} AS status,
                risk.description, risk.priority, risk.risk_level, risk.properties
            FROM {edges} r
            JOIN {entities} risk ON r.{tc.edge_source_column} = risk.{tc.node_id_column}
            JOIN entity_and_children target ON r.{tc.edge_target_column} = target.entity_id
            WHERE {risk_membership}
            ORDER BY CASE risk.{tc.node_status_column}
                WHEN 'materialized' THEN 1
                WHEN 'mitigating' THEN 2
                WHEN 'being_monitored' THEN 3
                WHEN 'identified' THEN 4
                ELSE 5
            END
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
        """Find people who own or are accountable for `entity_id`, owners before accountable
        parties — retains the source's relationship-type-priority ordering.

        Relies on the ontology's domain/range guarantee for `owns`/`accountable_for` (only a
        `Person`-typed entity can be their source) rather than an extra type literal filter.
        """
        tc = self._tc
        entities = self.dialect.qualify_table(tc.node_table)
        edges = self.dialect.qualify_table(tc.edge_table)
        ownership_rel_types = resolve_semantic(self.ontology, _OWNERSHIP_SEMANTIC)
        entity_param = self.dialect.param("entity_id")
        membership = self.dialect.array_membership(f"r.{tc.edge_type_column}", "rel_types")

        sql = f"""
            SELECT DISTINCT
                p.{tc.node_id_column} AS entity_id, p.{tc.node_name_column} AS name,
                p.{tc.node_type_column} AS type, p.description, p.properties,
                r.{tc.edge_type_column} AS relationship_type
            FROM {edges} r
            JOIN {entities} p ON r.{tc.edge_source_column} = p.{tc.node_id_column}
            WHERE r.{tc.edge_target_column} = {entity_param}
                AND {membership}
            ORDER BY CASE relationship_type
                WHEN 'owns' THEN 1
                WHEN 'accountable_for' THEN 2
                ELSE 3
            END
        """
        rows = self._execute(sql, {"entity_id": entity_id, "rel_types": ownership_rel_types})
        return [Entity(**{k: v for k, v in row.items() if k != "relationship_type"}) for row in rows]

    def traverse_relationships(
        self,
        start_id: str,
        rel_type: str | list[str],
        depth: int = 1,
        direction: str = "out",
    ) -> list[dict]:
        """Traverse relationship(s) from `start_id`.

        Raises ValueError for multi-hop "both" and, like the DuckDB backend (a deliberate
        simplification from the ported source, which supported single-hop "both" via an OR
        join), for single-hop "both" too — every backend behaves identically here.
        """
        tc = self._tc
        entities = self.dialect.qualify_table(tc.node_table)
        edges = self.dialect.qualify_table(tc.edge_table)
        rel_types = [rel_type] if isinstance(rel_type, str) else rel_type
        start_param = self.dialect.param("start_id")
        membership = self.dialect.array_membership(f"r.{tc.edge_type_column}", "rel_types")

        if direction not in ("out", "in"):
            raise ValueError("direction must be 'out' or 'in' for this backend")

        if depth == 1:
            join_col, filter_col = (
                (tc.edge_target_column, tc.edge_source_column)
                if direction == "out"
                else (tc.edge_source_column, tc.edge_target_column)
            )
            sql = f"""
                SELECT DISTINCT
                    e.{tc.node_id_column} AS entity_id, e.{tc.node_name_column} AS name,
                    e.{tc.node_type_column} AS type, e.{tc.node_status_column} AS status,
                    r.{tc.edge_type_column} AS relationship_type, 1 AS depth
                FROM {edges} r
                JOIN {entities} e ON r.{join_col} = e.{tc.node_id_column}
                WHERE r.{filter_col} = {start_param}
                    AND {membership}
                LIMIT 100
            """
            return self._execute(sql, {"start_id": start_id, "rel_types": rel_types})

        outward_col = tc.edge_target_column if direction == "out" else tc.edge_source_column
        inward_col = tc.edge_source_column if direction == "out" else tc.edge_target_column
        depth_param = self.dialect.param("depth")

        sql = f"""
            WITH RECURSIVE traversal AS (
                SELECT
                    e.{tc.node_id_column} AS entity_id, e.{tc.node_name_column} AS name,
                    e.{tc.node_type_column} AS type, e.{tc.node_status_column} AS status,
                    r.{tc.edge_type_column} AS relationship_type, 1 AS depth
                FROM {edges} r
                JOIN {entities} e ON r.{outward_col} = e.{tc.node_id_column}
                WHERE r.{inward_col} = {start_param}
                    AND {membership}

                UNION ALL

                SELECT
                    e.{tc.node_id_column}, e.{tc.node_name_column},
                    e.{tc.node_type_column}, e.{tc.node_status_column},
                    r.{tc.edge_type_column}, t.depth + 1
                FROM traversal t
                JOIN {edges} r ON t.entity_id = r.{inward_col}
                JOIN {entities} e ON r.{outward_col} = e.{tc.node_id_column}
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
        entities = self.dialect.qualify_table(tc.node_table)
        name_param = self.dialect.param("name")
        select = (
            f"SELECT {tc.node_id_column} AS entity_id, {tc.node_name_column} AS name, "
            f"{tc.node_type_column} AS type, {tc.node_status_column} AS status, "
            f"description, priority, risk_level, properties FROM {entities}"
        )
        if exact:
            where = f"WHERE LOWER({tc.node_name_column}) = LOWER({name_param})"
        else:
            where = f"WHERE LOWER({tc.node_name_column}) LIKE LOWER(CONCAT('%', {name_param}, '%'))"

        params: dict[str, Any] = {"name": name}
        if entity_type:
            where += f" AND {tc.node_type_column} = {self.dialect.param('entity_type')}"
            params["entity_type"] = entity_type

        rows = self._execute(f"{select} {where} LIMIT 1", params)
        return Entity(**rows[0]) if rows else None

    def get_goals_status_summary(self) -> list[dict]:
        """Per-goal status plus RECURSIVE (transitive) initiative/project/task counts.

        Same deliberate improvement as the DuckDB backend over the ported source, which counted
        only direct `belongs_to` children — see `duckdb_backend.py` for why that undercounts on
        this repo's strict tree.
        """
        tc = self._tc
        entities = self.dialect.qualify_table(tc.node_table)
        edges = self.dialect.qualify_table(tc.edge_table)
        hierarchy_rel_types = resolve_semantic(self.ontology, _HIERARCHY_SEMANTIC)
        depth = effective_max_depth(self.ontology, hierarchy_rel_types, _DEFAULT_HIERARCHY_DEPTH)
        membership = self.dialect.array_membership(f"r.{tc.edge_type_column}", "rel_types")

        sql = f"""
            WITH RECURSIVE descendants AS (
                SELECT
                    g.{tc.node_id_column} AS goal_id,
                    e.{tc.node_id_column} AS entity_id, e.{tc.node_type_column} AS type,
                    1 AS depth
                FROM {entities} g
                JOIN {edges} r ON r.{tc.edge_target_column} = g.{tc.node_id_column}
                JOIN {entities} e ON r.{tc.edge_source_column} = e.{tc.node_id_column}
                WHERE {membership}

                UNION ALL

                SELECT d.goal_id, e.{tc.node_id_column}, e.{tc.node_type_column}, d.depth + 1
                FROM descendants d
                JOIN {edges} r ON r.{tc.edge_target_column} = d.entity_id
                JOIN {entities} e ON r.{tc.edge_source_column} = e.{tc.node_id_column}
                WHERE {membership}
                    AND d.depth < {int(depth)}
            )
            SELECT
                g.{tc.node_id_column} AS entity_id, g.{tc.node_name_column} AS name,
                g.{tc.node_status_column} AS status, g.description,
                COUNT(DISTINCT CASE WHEN d.type = 'Initiative' THEN d.entity_id END) AS initiative_count,
                COUNT(DISTINCT CASE WHEN d.type = 'Project' THEN d.entity_id END) AS project_count,
                COUNT(DISTINCT CASE WHEN d.type = 'Task' THEN d.entity_id END) AS task_count
            FROM {entities} g
            LEFT JOIN descendants d ON g.{tc.node_id_column} = d.goal_id
            WHERE g.{tc.node_type_column} = 'Goal'
            GROUP BY g.{tc.node_id_column}, g.{tc.node_name_column}, g.{tc.node_status_column}, g.description
            ORDER BY name
        """
        return self._execute(sql, {"rel_types": hierarchy_rel_types})
