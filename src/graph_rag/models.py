"""Runtime data models for entities, relationships, and query result shapes.

Entity and relationship type fields are plain `str`, validated against a loaded `Ontology` at
the point of use (see `graph_rag.ontology.resolve`) rather than hardcoded as enums — enums are
exactly the coupling the ontology registry exists to remove.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel, field_serializer, field_validator


class Entity(BaseModel):
    """A graph node."""

    entity_id: str
    name: str
    type: str
    status: str | None = None
    owner_id: str | None = None
    description: str | None = None
    priority: str | None = None
    risk_level: str | None = None
    properties: dict[str, Any] = {}
    created_at: datetime
    updated_at: datetime | None = None

    @field_validator("properties", mode="before")
    @classmethod
    def _parse_properties(cls, value: object) -> dict[str, Any]:
        if isinstance(value, str):
            return json.loads(value) if value else {}
        return value or {}

    @field_serializer("properties")
    def _serialize_properties(self, value: dict[str, Any]) -> str:
        return json.dumps(value, sort_keys=True)


class Relationship(BaseModel):
    """A graph edge."""

    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    confidence: float = 1.0
    created_at: datetime


class BlockerHit(BaseModel):
    """One entity found while walking a transitive chain (e.g. blockers) back to a target."""

    entity_id: str
    name: str
    status: str | None = None
    distance: int
    rel_chain: list[str]


class EnrichmentResult(BaseModel):
    """The batch-enrichment groups for a single entity: hierarchy, blockers, risks, owners."""

    entity_id: str
    hierarchy: list[Entity] = []
    blockers: list[BlockerHit] = []
    risks: list[Entity] = []
    owners: list[Entity] = []
