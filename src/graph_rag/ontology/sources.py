"""Pluggable sources for loading an Ontology.

Every source hydrates a complete `Ontology` object up front — never a live handle, cursor, or
lazy proxy. That contract is what lets a table-backed or HTTP-backed source drop in later with
zero changes to any caller.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import yaml

from graph_rag.ontology.models import (
    EntityTypeDef,
    Ontology,
    RelationshipTypeDef,
    Semantic,
    TableConfig,
)

if TYPE_CHECKING:
    from graph_rag.dialects.base import SqlDialect

# A callable that runs SQL with named parameters and returns rows as dicts — the one seam
# `TableOntologySource` needs from a storage engine. Deliberately narrower than `GraphBackend`:
# reading three small config tables at startup has nothing to do with graph traversal.
QueryExecutor = Callable[[str, dict[str, Any]], list[dict[str, Any]]]


@runtime_checkable
class OntologySource(Protocol):
    """A source that can produce a fully hydrated, validated Ontology."""

    def load(self) -> Ontology:
        """Load and validate the ontology, returning a complete in-memory object."""
        ...


class FileOntologySource:
    """Loads an Ontology from a YAML file on disk."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> Ontology:
        """Read and parse the YAML file, then validate it into an Ontology."""
        with self.path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return Ontology.model_validate(raw)


class TableOntologySource:
    """Loads an Ontology from database tables — the "dynamic" proof.

    Reads `ontology_entity_types`, `ontology_relationship_types`, and `ontology_semantics`
    (see `sql/ontology_tables.sql`) for the given `ontology_name` and hydrates them into the
    same `Ontology` shape `FileOntologySource` produces. A caller cannot tell the two apart —
    that equality is asserted directly in `tests/test_ontology.py`.

    `version` and `table_config` are physical-deployment configuration (which columns store
    the graph, what version string to stamp), not part of the dynamic vocabulary these tables
    hold, so they're supplied by the caller rather than read from a table. What IS dynamic —
    entity types, relationship types (including which are transitive, their depth caps), and
    semantic aliases — is a live source of truth: adding a relationship type is an INSERT, not
    a redeploy. See `demo.py`'s live-swap section for that in action.

    Args:
        execute: Runs a parameterized SQL query and returns rows as dicts.
        dialect: Supplies this engine's parameter placeholder and table-qualification syntax.
        ontology_name: Which ontology's rows to load (tables can hold more than one).
        version: Version string to stamp on the hydrated Ontology.
        table_config: Physical table/column names backing the graph itself.
    """

    def __init__(
        self,
        execute: QueryExecutor,
        dialect: SqlDialect,
        ontology_name: str,
        version: str,
        table_config: TableConfig,
    ) -> None:
        self.execute = execute
        self.dialect = dialect
        self.ontology_name = ontology_name
        self.version = version
        self.table_config = table_config

    def load(self) -> Ontology:
        """Query the ontology tables and hydrate a fully validated Ontology.

        Rows are ordered by `seq`, an explicit ordinal column, rather than by name — a plain
        `ORDER BY name` would silently re-sort entity/relationship types alphabetically, which
        would NOT match a YAML source's declared order and would break the equality check
        `FileOntologySource(...).load() == TableOntologySource(...).load()` (list fields on a
        pydantic model compare order-sensitively) even when the two sources hold the same
        vocabulary.
        """
        name_param = self.dialect.param("ontology_name")
        params = {"ontology_name": self.ontology_name}

        entity_rows = self.execute(
            f"SELECT name, description "
            f"FROM {self.dialect.qualify_table('ontology_entity_types')} "
            f"WHERE ontology_name = {name_param} ORDER BY seq",
            params,
        )
        entity_types = [EntityTypeDef(**row) for row in entity_rows]

        rel_rows = self.execute(
            f"SELECT name, description, source_types, target_types, inverse, traversal, "
            f"canonical_direction, max_depth, fan_out_limit "
            f"FROM {self.dialect.qualify_table('ontology_relationship_types')} "
            f"WHERE ontology_name = {name_param} ORDER BY seq",
            params,
        )
        relationship_types = [
            RelationshipTypeDef(
                name=row["name"],
                description=row["description"],
                source_types=_load_list(row["source_types"]),
                target_types=_load_list(row["target_types"]),
                inverse=row["inverse"],
                traversal=row["traversal"],
                canonical_direction=row["canonical_direction"],
                max_depth=row["max_depth"],
                fan_out_limit=row["fan_out_limit"],
            )
            for row in rel_rows
        ]

        sem_rows = self.execute(
            f"SELECT name, relationship_types "
            f"FROM {self.dialect.qualify_table('ontology_semantics')} "
            f"WHERE ontology_name = {name_param} ORDER BY seq",
            params,
        )
        semantics = [
            Semantic(name=row["name"], relationship_types=_load_list(row["relationship_types"]))
            for row in sem_rows
        ]

        return Ontology(
            name=self.ontology_name,
            version=self.version,
            entity_types=entity_types,
            relationship_types=relationship_types,
            semantics=semantics,
            table_config=self.table_config,
        )


def _load_list(value: Any) -> list[str]:
    """Parse a JSON-array-string column, tolerating a value already deserialized to a list."""
    return json.loads(value) if isinstance(value, str) else list(value)
