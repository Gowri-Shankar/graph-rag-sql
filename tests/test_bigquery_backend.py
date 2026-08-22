"""Fully mocked tests for BigQueryGraphBackend — no network, no credentials.

`google-cloud-bigquery` is not required to run this suite: when it isn't installed, a fake
`google.cloud.bigquery` module is injected into `sys.modules` before constructing the backend,
so the lazy `from google.cloud import bigquery` import inside `__init__` resolves to the fake.
"""

from __future__ import annotations

import sys
import types

import pytest

from graph_rag.ontology import FileOntologySource, Ontology

ORG_GRAPH_PATH = "ontology/org_graph.yaml"


def _make_fake_bigquery_module() -> types.ModuleType:
    """A minimal stand-in for `google.cloud.bigquery`: just enough surface for
    `BigQueryGraphBackend` to construct query parameters and a job config.
    """
    fake_bigquery = types.ModuleType("google.cloud.bigquery")

    class ScalarQueryParameter:
        def __init__(self, name, type_, value):
            self.name = name
            self.type_ = type_
            self.value = value

    class ArrayQueryParameter:
        def __init__(self, name, array_type, values):
            self.name = name
            self.array_type = array_type
            self.values = list(values)

    class QueryJobConfig:
        def __init__(self, query_parameters=None):
            self.query_parameters = query_parameters or []

    class Client:
        def __init__(self, project=None):
            self.project = project

        def query(self, sql, job_config=None):  # pragma: no cover - overridden per test
            raise NotImplementedError("test must set backend.client.query")

    fake_bigquery.ScalarQueryParameter = ScalarQueryParameter
    fake_bigquery.ArrayQueryParameter = ArrayQueryParameter
    fake_bigquery.QueryJobConfig = QueryJobConfig
    fake_bigquery.Client = Client
    return fake_bigquery


class FakeQueryJob:
    """Stands in for a `bigquery.QueryJob`: `.result()` returns pre-canned rows."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def result(self):
        return self._rows


@pytest.fixture
def fake_bigquery(monkeypatch):
    """Install a fake `google.cloud.bigquery` module, whether or not the real one is present."""
    fake = _make_fake_bigquery_module()
    fake_google = types.ModuleType("google")
    fake_cloud = types.ModuleType("google.cloud")
    fake_google.cloud = fake_cloud
    fake_cloud.bigquery = fake

    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.cloud", fake_cloud)
    monkeypatch.setitem(sys.modules, "google.cloud.bigquery", fake)
    return fake


@pytest.fixture
def org_ontology() -> Ontology:
    return Ontology.from_source(FileOntologySource(ORG_GRAPH_PATH))


def _make_backend(fake_bigquery, org_ontology):
    from graph_rag.backends.bigquery_backend import BigQueryGraphBackend

    return BigQueryGraphBackend(org_ontology, project_id="test-project", dataset_id="test_dataset")


def test_constructor_raises_without_bigquery_installed(monkeypatch, org_ontology):
    """Without the [bigquery] extra installed, construction fails with a helpful ImportError."""
    monkeypatch.delitem(sys.modules, "google.cloud.bigquery", raising=False)
    monkeypatch.delitem(sys.modules, "google.cloud", raising=False)
    monkeypatch.delitem(sys.modules, "google", raising=False)

    try:
        import google.cloud.bigquery  # noqa: F401

        pytest.skip("google-cloud-bigquery is actually installed in this environment")
    except ImportError:
        pass

    from graph_rag.backends.bigquery_backend import BigQueryGraphBackend

    with pytest.raises(ImportError, match=r"graph-rag-sql\[bigquery\]"):
        BigQueryGraphBackend(org_ontology, project_id="p", dataset_id="d")


def test_constructor_raises_without_project_or_dataset(fake_bigquery, monkeypatch, org_ontology):
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)
    monkeypatch.delenv("BQ_DATASET_ID", raising=False)
    from graph_rag.backends.bigquery_backend import BigQueryGraphBackend

    with pytest.raises(ValueError, match="GCP_PROJECT_ID"):
        BigQueryGraphBackend(org_ontology)


def test_constructor_falls_back_to_env_vars(fake_bigquery, monkeypatch, org_ontology):
    monkeypatch.setenv("GCP_PROJECT_ID", "env-project")
    monkeypatch.setenv("BQ_DATASET_ID", "env_dataset")
    from graph_rag.backends.bigquery_backend import BigQueryGraphBackend

    backend = BigQueryGraphBackend(org_ontology)
    assert backend.project_id == "env-project"
    assert backend.dataset_id == "env_dataset"


def test_find_blockers_parameterizes_rel_types(fake_bigquery, org_ontology):
    """Regression test for the injection fix: relationship types must travel as a genuine
    ArrayQueryParameter through `IN UNNEST(...)`, never joined into the SQL text.
    """
    backend = _make_backend(fake_bigquery, org_ontology)
    captured = {}

    def fake_query(sql, job_config=None):
        captured["sql"] = sql
        captured["job_config"] = job_config
        return FakeQueryJob([])

    backend.client.query = fake_query
    backend.find_blockers("proj-atlas", max_depth=3)

    assert "WITH RECURSIVE" in captured["sql"]
    assert "IN UNNEST(@rel_types)" in captured["sql"]
    # Regression guard for the injection fix: the source built this filter by joining type
    # names into a quoted, comma-separated string and interpolating it into an IN (...) clause.
    assert "IN ('blocks'" not in captured["sql"]

    array_params = [p for p in captured["job_config"].query_parameters if p.name == "rel_types"]
    assert len(array_params) == 1
    assert array_params[0].array_type == "STRING"
    assert set(array_params[0].values) == {"blocks", "depends_on"}


def test_traverse_relationships_parameterizes_rel_types(fake_bigquery, org_ontology):
    backend = _make_backend(fake_bigquery, org_ontology)
    captured = {}

    def fake_query(sql, job_config=None):
        captured["sql"] = sql
        captured["job_config"] = job_config
        return FakeQueryJob([])

    backend.client.query = fake_query
    backend.traverse_relationships("proj-atlas", ["blocks", "depends_on"], depth=3, direction="in")

    assert "WITH RECURSIVE" in captured["sql"]
    assert "IN UNNEST(@rel_types)" in captured["sql"]
    array_params = [p for p in captured["job_config"].query_parameters if p.name == "rel_types"]
    assert len(array_params) == 1
    assert array_params[0].array_type == "STRING"


def test_find_blockers_returns_typed_blocker_hits(fake_bigquery, org_ontology):
    backend = _make_backend(fake_bigquery, org_ontology)
    rows = [
        {
            "entity_id": "task-1",
            "name": "Task One",
            "status": "blocked",
            "distance": 1,
            "rel_chain": ["blocks"],
            "name_chain": ["Task One"],
        }
    ]
    backend.client.query = lambda sql, job_config=None: FakeQueryJob(rows)

    hits = backend.find_blockers("proj-atlas", max_depth=3)
    assert len(hits) == 1
    assert hits[0].entity_id == "task-1"
    assert hits[0].distance == 1


def test_enrich_entities_batch_parses_nested_properties_and_defaults(fake_bigquery, org_ontology):
    backend = _make_backend(fake_bigquery, org_ontology)

    class StructRow(dict):
        """Stands in for a BigQuery nested STRUCT row: dict-like via `.items()`."""

    rows = [
        {
            "entity_id": "proj-atlas",
            "hierarchy": [StructRow(entity_id="init-1", name="Init One", type="Initiative", status="in_progress", properties='{"a": 1}')],
            "blockers": [],
            "risks": [],
            "owners": [StructRow(entity_id="person-1", name="Person One", type="Person", status=None, properties="{}")],
        }
    ]
    backend.client.query = lambda sql, job_config=None: FakeQueryJob(rows)

    result = backend.enrich_entities_batch(["proj-atlas"])
    assert set(result) == {"proj-atlas"}
    enrichment = result["proj-atlas"]
    assert len(enrichment.hierarchy) == 1
    assert enrichment.hierarchy[0].properties == {"a": 1}
    assert enrichment.blockers == []
    assert enrichment.risks == []
    assert len(enrichment.owners) == 1


def test_enrich_entities_batch_returns_empty_dict_on_client_error(fake_bigquery, org_ontology):
    backend = _make_backend(fake_bigquery, org_ontology)

    def raising_query(sql, job_config=None):
        raise RuntimeError("simulated BigQuery outage")

    backend.client.query = raising_query
    assert backend.enrich_entities_batch(["proj-atlas"]) == {}


def test_enrich_entities_batch_empty_input_short_circuits(fake_bigquery, org_ontology):
    backend = _make_backend(fake_bigquery, org_ontology)
    backend.client.query = lambda sql, job_config=None: (_ for _ in ()).throw(
        AssertionError("should not query for an empty batch")
    )
    assert backend.enrich_entities_batch([]) == {}


def test_get_entity_owners_qualifies_table_name(fake_bigquery, org_ontology):
    backend = _make_backend(fake_bigquery, org_ontology)
    captured = {}

    def fake_query(sql, job_config=None):
        captured["sql"] = sql
        return FakeQueryJob([])

    backend.client.query = fake_query
    backend.get_entity_owners("proj-atlas")
    assert "`test-project.test_dataset.canonical_entities`" in captured["sql"]
    assert "`test-project.test_dataset.entity_relationships`" in captured["sql"]
