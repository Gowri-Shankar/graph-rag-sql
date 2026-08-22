-- graph-rag-sql: BigQuery DDL for the two-table property graph.
--
-- Adapted (and deliberately simplified) from a production knowledge-graph system's schema.
-- Dropped columns that aren't needed for graph traversal: aliases, embeddings, provenance
-- arrays, mention counts, progress percentages, and similar extraction-pipeline bookkeeping.
-- The full entity/relationship shape lives in graph_rag.models; this file only needs the
-- columns the recursive-CTE patterns actually read.

-- ==========================================
-- Nodes: canonical_entities
-- ==========================================
CREATE OR REPLACE TABLE `dataset.canonical_entities` (
    entity_id STRING NOT NULL OPTIONS(description="Stable unique identifier for this entity"),
    name STRING NOT NULL OPTIONS(description="Canonical display name of the entity"),

    type STRING NOT NULL OPTIONS(description="Entity type, validated against the ontology registry (e.g. Goal, Initiative, Project, Task, Person, Risk)"),
    status STRING OPTIONS(description="Current status, e.g. not_started, in_progress, at_risk, blocked, completed"),

    owner_id STRING OPTIONS(description="Email address of the owner/accountable party, e.g. owner@example.com"),

    description STRING OPTIONS(description="Free-text description of the entity"),
    priority STRING OPTIONS(description="Priority level: critical, high, medium, low"),
    risk_level STRING OPTIONS(description="Risk assessment: low, medium, high, critical"),

    properties STRING OPTIONS(description="JSON object storing type-specific attributes, serialized as a string"),

    created_at TIMESTAMP NOT NULL OPTIONS(description="When this record was created"),
    updated_at TIMESTAMP OPTIONS(description="When this record was last modified")
)
-- Partitioning by creation date lets time-bounded traversals and reloads prune partitions
-- instead of scanning the whole table; clustering by (type, status) matches the two filters
-- almost every pattern applies before it ever walks an edge.
PARTITION BY DATE(created_at)
CLUSTER BY type, status
OPTIONS(
    description="Canonical entity nodes for the graph — one row per node, entity_id is the join key entity_relationships uses"
);

-- ==========================================
-- Edges: entity_relationships
-- ==========================================
CREATE OR REPLACE TABLE `dataset.entity_relationships` (
    source_entity_id STRING NOT NULL OPTIONS(description="Source entity_id (from canonical_entities)"),
    target_entity_id STRING NOT NULL OPTIONS(description="Target entity_id (from canonical_entities)"),
    relationship_type STRING NOT NULL OPTIONS(description="Edge type, validated against the ontology registry: belongs_to, blocks, depends_on, owns, accountable_for, threatens"),

    confidence FLOAT64 OPTIONS(description="0.0-1.0 confidence score for this edge (default 1.0 for hand-authored/synthetic data)"),

    created_at TIMESTAMP NOT NULL OPTIONS(description="When this relationship was created")
)
-- Every recursive traversal expands one row's outgoing (or incoming) edges by
-- relationship_type, so clustering on (source_entity_id, relationship_type) puts the exact
-- rows each recursive step needs next to each other on disk. Partitioning by created_at
-- keeps that clustering cheap to maintain as new edges are appended over time.
PARTITION BY DATE(created_at)
CLUSTER BY source_entity_id, relationship_type
OPTIONS(
    description="Directed edges connecting canonical_entities nodes. Logical unique key: (source_entity_id, target_entity_id, relationship_type)"
);
