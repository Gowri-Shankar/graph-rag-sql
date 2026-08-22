"""Pluggable sources for loading an Ontology.

Every source hydrates a complete `Ontology` object up front — never a live handle, cursor, or
lazy proxy. That contract is what lets a table-backed or HTTP-backed source drop in later with
zero changes to any caller.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import yaml

from graph_rag.ontology.models import Ontology


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
    """Loads an Ontology from a database table.

    Not implemented in this milestone. A later milestone backs this with a table so that
    adding a traversable relationship type becomes an INSERT rather than a redeploy — the
    `OntologySource` Protocol means no caller code changes when this lands.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError("TableOntologySource is implemented in a later milestone")

    def load(self) -> Ontology:
        raise NotImplementedError("TableOntologySource is implemented in a later milestone")
