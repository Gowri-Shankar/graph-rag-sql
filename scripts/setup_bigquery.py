"""One-command loader for the BigQuery sandbox: dataset, tables, ontology, and demo data.

The BigQuery sandbox (https://cloud.google.com/bigquery/docs/sandbox) works with just a Google
account — no billing account, no credit card. This script creates the dataset if it doesn't
exist, applies `sql/bigquery_schema.sql`'s two-table DDL and `sql/ontology_tables.sql`'s
three-table DDL (rendered with the real project/dataset), loads `data/*.csv` with an explicit
schema, and seeds the ontology tables from `ontology/org_graph.yaml` — so the BigQuery path can
run with either an `OntologySource`.

Usage:
    On Windows PowerShell, load `.env` into the session first:
        Get-Content .env | ForEach-Object { if ($_ -match '^([^#=]+)=(.*)$') {
            [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2]) } }
    On bash/zsh:
        set -a; source .env; set +a
    Then run:
        python scripts/setup_bigquery.py

Requires GCP_PROJECT_ID and BQ_DATASET_ID in the environment (see `.env.example`), and the
`[bigquery]` extra installed: `pip install -e ".[bigquery]"`.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def _load_entities(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_relationships(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["confidence"] = float(row["confidence"])
    return rows


def main() -> None:
    try:
        from google.cloud import bigquery
    except ImportError as exc:
        raise SystemExit(
            'google-cloud-bigquery is not installed. Run: pip install -e ".[bigquery]"'
        ) from exc

    project_id = _require_env("GCP_PROJECT_ID")
    dataset_id = _require_env("BQ_DATASET_ID")

    client = bigquery.Client(project=project_id)
    dataset_ref = bigquery.DatasetReference(project_id, dataset_id)
    client.create_dataset(bigquery.Dataset(dataset_ref), exists_ok=True)
    print(f"Dataset ready: {project_id}.{dataset_id}")

    _apply_schema(client, project_id, dataset_id, "bigquery_schema.sql")
    _apply_schema(client, project_id, dataset_id, "ontology_tables.sql")

    entities = _load_entities(REPO_ROOT / "data" / "entities.csv")
    relationships = _load_relationships(REPO_ROOT / "data" / "relationships.csv")

    _load_table(
        client,
        dataset_ref,
        "canonical_entities",
        entities,
        [
            bigquery.SchemaField("entity_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("name", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("type", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("status", "STRING"),
            bigquery.SchemaField("owner_id", "STRING"),
            bigquery.SchemaField("description", "STRING"),
            bigquery.SchemaField("priority", "STRING"),
            bigquery.SchemaField("risk_level", "STRING"),
            bigquery.SchemaField("properties", "STRING"),
            bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("updated_at", "TIMESTAMP"),
        ],
    )
    _load_table(
        client,
        dataset_ref,
        "entity_relationships",
        relationships,
        [
            bigquery.SchemaField("source_entity_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("target_entity_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("relationship_type", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("confidence", "FLOAT64"),
            bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
        ],
    )

    _seed_ontology_tables(client, dataset_ref, project_id, dataset_id)

    print("Setup complete.")


def _apply_schema(client, project_id: str, dataset_id: str, filename: str) -> None:
    """Run each `CREATE OR REPLACE TABLE` statement in a schema file, substituting the real
    project/dataset for the placeholder `dataset.` prefix the checked-in SQL files use.
    """
    sql_text = (REPO_ROOT / "sql" / filename).read_text(encoding="utf-8")
    sql_text = sql_text.replace("`dataset.", f"`{project_id}.{dataset_id}.")
    for statement in _split_statements(sql_text):
        client.query(statement).result()
    print(f"Applied {filename}")


def _split_statements(sql_text: str) -> list[str]:
    statements = []
    for raw_statement in sql_text.split(";"):
        lines = [line for line in raw_statement.splitlines() if not line.strip().startswith("--")]
        statement = "\n".join(lines).strip()
        if statement:
            statements.append(statement)
    return statements


def _load_table(client, dataset_ref, table_name: str, rows: list[dict], schema: list) -> None:
    from google.cloud import bigquery

    table_ref = dataset_ref.table(table_name)
    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition="WRITE_TRUNCATE",
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
    )
    load_job = client.load_table_from_json(rows, table_ref, job_config=job_config)
    load_job.result()
    print(f"Loaded {len(rows)} rows into {table_name}")


def _seed_ontology_tables(client, dataset_ref, project_id: str, dataset_id: str) -> None:
    """Populate the ontology_* tables from `ontology/org_graph.yaml`.

    Lets the BigQuery path run against either `FileOntologySource` (reading the YAML directly)
    or `TableOntologySource` (reading these rows) for the exact same vocabulary.
    """
    from google.cloud import bigquery

    from graph_rag.ontology import FileOntologySource, Ontology

    ontology = Ontology.from_source(FileOntologySource(str(REPO_ROOT / "ontology" / "org_graph.yaml")))

    entity_type_rows = [
        {"ontology_name": ontology.name, "seq": i, "name": et.name, "description": et.description}
        for i, et in enumerate(ontology.entity_types)
    ]
    rel_type_rows = [
        {
            "ontology_name": ontology.name,
            "seq": i,
            "name": rt.name,
            "description": rt.description,
            "source_types": json.dumps(rt.source_types),
            "target_types": json.dumps(rt.target_types),
            "inverse": rt.inverse,
            "traversal": rt.traversal,
            "canonical_direction": rt.canonical_direction,
            "max_depth": rt.max_depth,
            "fan_out_limit": rt.fan_out_limit,
        }
        for i, rt in enumerate(ontology.relationship_types)
    ]
    semantic_rows = [
        {
            "ontology_name": ontology.name,
            "seq": i,
            "name": sem.name,
            "relationship_types": json.dumps(sem.relationship_types),
        }
        for i, sem in enumerate(ontology.semantics)
    ]

    _load_table(
        client,
        dataset_ref,
        "ontology_entity_types",
        entity_type_rows,
        [
            bigquery.SchemaField("ontology_name", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("seq", "INT64", mode="REQUIRED"),
            bigquery.SchemaField("name", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("description", "STRING", mode="REQUIRED"),
        ],
    )
    _load_table(
        client,
        dataset_ref,
        "ontology_relationship_types",
        rel_type_rows,
        [
            bigquery.SchemaField("ontology_name", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("seq", "INT64", mode="REQUIRED"),
            bigquery.SchemaField("name", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("description", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("source_types", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("target_types", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("inverse", "STRING"),
            bigquery.SchemaField("traversal", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("canonical_direction", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("max_depth", "INT64"),
            bigquery.SchemaField("fan_out_limit", "INT64"),
        ],
    )
    _load_table(
        client,
        dataset_ref,
        "ontology_semantics",
        semantic_rows,
        [
            bigquery.SchemaField("ontology_name", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("seq", "INT64", mode="REQUIRED"),
            bigquery.SchemaField("name", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("relationship_types", "STRING", mode="REQUIRED"),
        ],
    )


if __name__ == "__main__":
    main()
