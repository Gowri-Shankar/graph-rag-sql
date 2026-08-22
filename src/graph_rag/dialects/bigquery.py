"""`SqlDialect` implementation for BigQuery.

This is the only file in the repo that should contain BigQuery-specific SQL syntax. Each
method mirrors `dialects/duckdb.py` member-for-member — see that file's docstrings for the
DuckDB spelling each of these stands in for.

This module intentionally does NOT import `google.cloud.bigquery` — it only renders SQL text
and table names, so it carries no dependency on the `[bigquery]` extra. `bigquery_backend.py`
is the file that imports the client library, and does so lazily.
"""

from __future__ import annotations

from typing import Any

from graph_rag.dialects.base import SqlDialect


class BigQueryDialect(SqlDialect):
    """Renders `SqlDialect` operations as BigQuery SQL."""

    def __init__(self, project_id: str, dataset_id: str) -> None:
        self.project_id = project_id
        self.dataset_id = dataset_id

    def param(self, name: str) -> str:
        """BigQuery named parameters are spelled `@name` (DuckDB: `$name`)."""
        return f"@{name}"

    def bind(self, params: dict[str, Any]) -> Any:
        """Unused by the BigQuery backend: query parameters need per-value type information
        (STRING/INT64/ARRAY) that a plain dict can't carry, so `BigQueryGraphBackend` builds
        `bigquery.QueryJobConfig` directly from the params dict instead of going through this.
        """
        return params

    def qualify_table(self, table_name: str) -> str:
        """BigQuery needs a backticked `project.dataset.table` reference (DuckDB: a bare name)."""
        return f"`{self.project_id}.{self.dataset_id}.{table_name}`"

    def array_literal(self, expressions: list[str]) -> str:
        """BigQuery (like DuckDB) spells an array constructor with square brackets."""
        return "[" + ", ".join(expressions) + "]"

    def array_append(self, array_expr: str, item_array_expr: str) -> str:
        """BigQuery concatenates arrays with `ARRAY_CONCAT(a, b)` (DuckDB: `a || b`)."""
        return f"ARRAY_CONCAT({array_expr}, {item_array_expr})"

    def array_membership(self, column_expr: str, param_name: str) -> str:
        """BigQuery spells array membership `IN UNNEST(@x)` (DuckDB: `IN (SELECT UNNEST($x))`)."""
        return f"{column_expr} IN UNNEST({self.param(param_name)})"

    def struct_agg(self, alias: str, fields: list[str], limit: int | None = None) -> str:
        """BigQuery struct aggregation: `ARRAY_AGG(STRUCT(a, b, ...) [LIMIT n])`.

        Unlike DuckDB's `list(...)`, BigQuery supports a per-group `LIMIT` directly inside the
        aggregate call, so a requested `limit` never needs a post-aggregation slice.
        """
        struct_expr = "STRUCT(" + ", ".join(fields) + ")"
        limit_clause = f" LIMIT {int(limit)}" if limit is not None else ""
        return f"ARRAY_AGG({struct_expr}{limit_clause}) AS {alias}"

    def empty_array(self, struct_fields: list[tuple[str, str]]) -> str:
        """BigQuery needs no typed empty-array literal: `COALESCE(aggregate, [])` already infers
        the array's struct type from the aggregate expression itself (DuckDB requires an
        explicit `CAST([] AS STRUCT(...)[])` because its `COALESCE` cannot infer it).
        """
        return "[]"

    def top_n_per_group(self, partition_by: list[str], order_by: str, n: int) -> str:
        """Both BigQuery and DuckDB support `QUALIFY` (Postgres does not — hence the Protocol)."""
        partitions = ", ".join(partition_by)
        return f"QUALIFY ROW_NUMBER() OVER (PARTITION BY {partitions} ORDER BY {order_by}) <= {n}"
