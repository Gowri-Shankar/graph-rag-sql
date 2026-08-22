"""`SqlDialect` implementation for DuckDB.

This is the only file in the repo that should contain DuckDB-specific SQL syntax. Each method
documents the BigQuery spelling it stands in for, so milestone 3's `dialects/bigquery.py` can
be written by mirroring this file member-for-member.
"""

from __future__ import annotations

from typing import Any, ClassVar

from graph_rag.dialects.base import SqlDialect


class DuckDbDialect(SqlDialect):
    """Renders `SqlDialect` operations as DuckDB SQL."""

    _TYPE_MAP: ClassVar[dict[str, str]] = {
        "string": "VARCHAR", "int": "BIGINT", "string[]": "VARCHAR[]"
    }

    def param(self, name: str) -> str:
        """DuckDB named parameters are spelled `$name` (BigQuery: `@name`)."""
        return f"${name}"

    def bind(self, params: dict[str, Any]) -> Any:
        """DuckDB's `execute()` accepts a plain name->value dict directly."""
        return params

    def qualify_table(self, table_name: str) -> str:
        """DuckDB reads a plain table name (BigQuery needs backticked `project.dataset.table`)."""
        return table_name

    def array_literal(self, expressions: list[str]) -> str:
        """DuckDB (like BigQuery) spells an array constructor with square brackets."""
        return "[" + ", ".join(expressions) + "]"

    def array_append(self, array_expr: str, item_array_expr: str) -> str:
        """DuckDB concatenates lists with `||` (BigQuery: `ARRAY_CONCAT(a, b)`)."""
        return f"({array_expr} || {item_array_expr})"

    def array_membership(self, column_expr: str, param_name: str) -> str:
        """DuckDB spells array membership `IN (SELECT UNNEST($x))` (BigQuery: `IN UNNEST(@x)`)."""
        return f"{column_expr} IN (SELECT UNNEST({self.param(param_name)}))"

    def struct_agg(self, alias: str, fields: list[str], limit: int | None = None) -> str:
        """DuckDB struct aggregation: `list({'a': a, 'b': b, ...})` (BigQuery: `ARRAY_AGG(STRUCT(...))`).

        DuckDB has no per-group `LIMIT` inside an aggregate call the way BigQuery's
        `ARRAY_AGG(... LIMIT n)` does, so a requested `limit` is applied by slicing the
        aggregated list afterward (DuckDB list slices are 1-indexed and inclusive).
        """
        struct_expr = "{" + ", ".join(f"'{f}': {f}" for f in fields) + "}"
        agg = f"list({struct_expr})"
        if limit is not None:
            agg = f"({agg})[1:{limit}]"
        return f"{agg} AS {alias}"

    def empty_array(self, struct_fields: list[tuple[str, str]]) -> str:
        """DuckDB spells a typed empty array `CAST([] AS STRUCT(...)[])`.

        BigQuery does not need this — `COALESCE(aggregate, [])` infers the type from the
        aggregate's own struct shape.
        """
        fields = ", ".join(f"{name} {self._TYPE_MAP[kind]}" for name, kind in struct_fields)
        return f"CAST([] AS STRUCT({fields})[])"

    def top_n_per_group(self, partition_by: list[str], order_by: str, n: int) -> str:
        """Both DuckDB and BigQuery support `QUALIFY` (Postgres does not — hence the Protocol)."""
        partitions = ", ".join(partition_by)
        return f"QUALIFY ROW_NUMBER() OVER (PARTITION BY {partitions} ORDER BY {order_by}) <= {n}"
