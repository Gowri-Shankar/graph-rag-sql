"""`SqlDialect`: the Protocol that absorbs SQL-engine differences.

A warehouse-portable graph query written with plain `WITH RECURSIVE` needs only about eight
engine-specific spellings beyond standard SQL: how a bound parameter looks, how a table is
qualified, how array literals/concatenation/membership are spelled, how struct aggregation is
spelled, how a typed empty array is spelled, and how "top N per group" is spelled (`QUALIFY`
is not universal — Postgres has no such clause). Everything else is ordinary SQL and belongs
directly in a pattern module.

Only add a member here when two real engines spell that operation differently. That keeps
"add a backend" equal to "write one small adapter class" rather than "touch every pattern".
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SqlDialect(Protocol):
    """Engine-specific SQL rendering, injected into pattern modules."""

    def param(self, name: str) -> str:
        """Render a bound-parameter placeholder for `name` (e.g. ``$name``, ``@name``)."""
        ...

    def bind(self, params: dict[str, Any]) -> Any:
        """Convert a name->value dict into whatever this driver's execute() expects."""
        ...

    def qualify_table(self, table_name: str) -> str:
        """Render a fully qualified table reference for this engine."""
        ...

    def array_literal(self, expressions: list[str]) -> str:
        """Render an array constructed from SQL expressions, e.g. ``[r.relationship_type]``.

        `expressions` are raw SQL expression strings (column references or literals), not
        Python values — this is for building array literals *inside* rendered SQL text.
        """
        ...

    def array_append(self, array_expr: str, item_array_expr: str) -> str:
        """Render 'append the (already-array) `item_array_expr` onto `array_expr`'."""
        ...

    def array_membership(self, column_expr: str, param_name: str) -> str:
        """Render '`column_expr` is a member of the array bound to `param_name`'."""
        ...

    def struct_agg(self, alias: str, fields: list[str], limit: int | None = None) -> str:
        """Render '`AS alias`'-terminated aggregate expression collecting struct rows.

        Args:
            alias: Output column alias for the aggregated array-of-structs.
            fields: Column names to pack into each struct (also used as struct field names).
                Rendered bare and unqualified, so the caller is responsible for aggregating
                over a scope that already exposes exactly these names — alias a physical
                column list to them in an inner projection first.
            limit: Optional per-group cap on the number of aggregated rows.
        """
        ...

    def empty_array(self, struct_fields: list[tuple[str, str]]) -> str:
        """Render a typed empty array-of-structs literal, for `COALESCE(aggregate, <empty>)`.

        `struct_fields` is a list of `(field_name, generic_type)` pairs, where `generic_type`
        is one of `"string"`, `"int"`, or `"string[]"` — a small, engine-agnostic type
        vocabulary. Each dialect maps these onto its own concrete SQL type syntax, which is
        what keeps this call free of any engine-specific type name in a pattern module.
        """
        ...

    def top_n_per_group(self, partition_by: list[str], order_by: str, n: int) -> str:
        """Render a clause keeping only the top `n` rows per `partition_by` group."""
        ...
