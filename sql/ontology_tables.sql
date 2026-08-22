-- graph-rag-sql: BigQuery DDL for a table-backed ontology registry.
--
-- Mirrors the fields `graph_rag.ontology.models` validates (EntityTypeDef, RelationshipTypeDef,
-- Semantic) so `TableOntologySource` can hydrate an Ontology identical to what
-- `FileOntologySource` produces from `ontology/org_graph.yaml` for the same `ontology_name`.
-- List-valued fields (source_types, target_types, relationship_types) are stored as
-- JSON-encoded STRING columns rather than ARRAY<STRING> — the same portability choice this
-- repo already makes for `canonical_entities.properties` — so the same DDL and the same
-- `TableOntologySource` code work unchanged on any SQL engine, not just BigQuery.
--
-- `ontology_name` scopes every row so one deployment can host more than one vocabulary
-- (e.g. this repo's `org_graph` alongside `tests/fixtures/tiny_domain.yaml`'s vocabulary).
-- `seq` preserves declaration order: pydantic list fields compare order-sensitively, so
-- `TableOntologySource` orders by `seq` rather than `name` to stay equal to a YAML source that
-- doesn't declare its types alphabetically.

CREATE OR REPLACE TABLE `dataset.ontology_entity_types` (
    ontology_name STRING NOT NULL OPTIONS(description="Which ontology this row belongs to, e.g. 'org_graph'"),
    seq INT64 NOT NULL OPTIONS(description="Declaration order within this ontology, matching the YAML source's list order"),
    name STRING NOT NULL OPTIONS(description="Entity type name, e.g. 'Project'"),
    description STRING NOT NULL OPTIONS(description="Human-readable description of this entity type")
)
OPTIONS(
    description="Table-backed entity type declarations — the dynamic counterpart to an ontology YAML file's entity_types list"
);

CREATE OR REPLACE TABLE `dataset.ontology_relationship_types` (
    ontology_name STRING NOT NULL OPTIONS(description="Which ontology this row belongs to"),
    seq INT64 NOT NULL OPTIONS(description="Declaration order within this ontology"),
    name STRING NOT NULL OPTIONS(description="Relationship type name, e.g. 'blocks'"),
    description STRING NOT NULL OPTIONS(description="Human-readable description of this relationship type"),

    source_types STRING NOT NULL OPTIONS(description="JSON array of entity type names allowed as this edge's source"),
    target_types STRING NOT NULL OPTIONS(description="JSON array of entity type names allowed as this edge's target"),

    inverse STRING OPTIONS(description="Name of the inverse relationship type, if declared"),
    traversal STRING NOT NULL OPTIONS(description="'transitive' or 'terminal'"),
    canonical_direction STRING NOT NULL OPTIONS(description="'source_to_target' or 'target_to_source'"),
    max_depth INT64 OPTIONS(description="Hard cap on recursive traversal depth for this type; NULL means uncapped"),
    fan_out_limit INT64 OPTIONS(description="Optional cap on results returned per traversal step")
)
OPTIONS(
    description="Table-backed relationship type declarations. INSERT a new row here to make a new transitive relationship type traversable with no code change and no restart — see demo.py's live-swap section"
);

CREATE OR REPLACE TABLE `dataset.ontology_semantics` (
    ontology_name STRING NOT NULL OPTIONS(description="Which ontology this row belongs to"),
    seq INT64 NOT NULL OPTIONS(description="Declaration order within this ontology"),
    name STRING NOT NULL OPTIONS(description="Semantic alias name, e.g. 'upstream'"),
    relationship_types STRING NOT NULL OPTIONS(description="JSON array of concrete relationship type names this alias resolves to")
)
OPTIONS(
    description="Table-backed semantic aliases — the dynamic counterpart to an ontology YAML file's semantics list"
);
