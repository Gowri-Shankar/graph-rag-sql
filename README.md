# graph-rag-sql

[![CI](https://github.com/Gowri-Shankar/graph-rag-sql/actions/workflows/ci.yml/badge.svg)](https://github.com/Gowri-Shankar/graph-rag-sql/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Graph traversal for RAG **without a graph database** — recursive-CTE patterns on a SQL warehouse, inspired by a public engineering write-up on graph-DB-to-warehouse migrations.

Answers questions like **"What blocks Project Atlas?"** — with the full dependency *path*, not just the endpoints — using nothing but `WITH RECURSIVE` SQL on the same warehouse that already holds your data. Runs locally on DuckDB in seconds (zero cloud setup), and on BigQuery for the production path.

---

## The problem

RAG over organizational data needs two retrieval shapes:

1. **"Find things like X"** → vector search. Solved.
2. **"What blocks X?" / "Who owns the thing that blocks X?"** → relationship traversal. The standard answer is a dedicated graph database next to your vector store.

That standard answer has a hidden cost: **two storage systems that must agree**. Sync delays between them produce stale relationship data that surfaces to users as apparent hallucination — the model faithfully reports an edge that no longer exists. You now debug consistency instead of shipping features.

## The bet

For **bounded traversals (3–5 hops)** over data that already lives in a warehouse, you don't need a graph database at all. Recursive CTEs give you hierarchy walks, dependency chains, and batch enrichment in plain SQL — with **atomic consistency for free**, because the graph *is* the warehouse tables.

Prior art: [Why We Traded a Graph Database for BigQuery CTEs](https://www.latentview.com/blog/why-we-traded-a-graph-database-for-bigquery-ctes/) describes this trade-off playing out in production — a managed graph database replaced at roughly **94% lower cost (~$50/year vs ~$800/month)**, with end-to-end latency dominated by LLM generation, not retrieval.

## The three patterns

Each pattern lives in [`src/graph_rag/patterns/`](src/graph_rag/patterns/) as a documented module that renders dialect-specific SQL (DuckDB and BigQuery).

### ① Bounded recursive hierarchy

Walk `belongs_to` edges up (ancestors) or down (subtree), depth-capped so cycles and fan-out can't hurt you.

Rendered by `patterns/hierarchy.py` for `direction="down"` — table names, column names and the
relationship-type list all come from the registry, which is why no type name appears as a
literal anywhere in it (dataset elided for width):

```sql
WITH RECURSIVE hierarchy AS (
  SELECT e.entity_id AS entity_id, e.name AS name,
         e.type AS type, e.status AS status,
         1 AS depth
  FROM entity_relationships r
  JOIN canonical_entities e ON r.source_entity_id = e.entity_id
  WHERE r.target_entity_id = @entity_id
    AND r.relationship_type IN UNNEST(@rel_types)

  UNION ALL

  SELECT e.entity_id AS entity_id, e.name AS name,
         e.type AS type, e.status AS status,
         h.depth + 1
  FROM hierarchy h
  JOIN entity_relationships r ON h.entity_id = r.target_entity_id
  JOIN canonical_entities e ON r.source_entity_id = e.entity_id
  WHERE r.relationship_type IN UNNEST(@rel_types)
    AND h.depth < 3          -- the bound is what makes this viable
)
SELECT * FROM hierarchy ORDER BY depth ASC;
```

```python
retriever.get_hierarchy("proj-atlas", direction="down")
```

### ② Blocker chains with path arrays (the showcase)

Follow `blocks` / `depends_on` edges transitively while **collecting the traversal path in arrays** — so the answer isn't "these 4 things block Atlas" but *how* each one blocks it.

```sql
WITH RECURSIVE blocker_chain AS (
 SELECT e.entity_id, e.name, e.status, 1 AS distance,
 [r.relationship_type] AS rel_chain,
 [e.name] AS name_chain
 FROM entity_relationships r
 JOIN canonical_entities e ON e.entity_id = r.source_entity_id
 WHERE r.target_entity_id = @entity_id
 AND r.relationship_type IN UNNEST(@rel_types)
 UNION ALL
 SELECT e.entity_id, e.name, e.status, bc.distance + 1,
 ARRAY_CONCAT(bc.rel_chain, [r.relationship_type]),
 ARRAY_CONCAT(bc.name_chain, [e.name])
 FROM blocker_chain bc
 JOIN entity_relationships r ON r.target_entity_id = bc.entity_id
 JOIN canonical_entities e ON e.entity_id = r.source_entity_id
 WHERE bc.distance < @max_depth
 AND r.relationship_type IN UNNEST(@rel_types)
)
SELECT * FROM blocker_chain
QUALIFY ROW_NUMBER() OVER (PARTITION BY entity_id, distance ORDER BY name) <= 1
ORDER BY distance ASC,
  CASE status WHEN 'blocked' THEN 1 WHEN 'at_risk' THEN 2 WHEN 'delayed' THEN 3 ELSE 4 END
LIMIT 50;
```

```python
retriever.find_blockers("proj-atlas", max_depth=5)
```

Two details worth noticing: relationship types travel as an **array query parameter** (`IN UNNEST(@rel_types)`) — never string-interpolated into SQL — and dedup happens via `QUALIFY` on scalar columns because BigQuery doesn't allow `SELECT DISTINCT` over ARRAY columns.

### ③ Batch enrichment via UNNEST (killing N+1)

After vector search returns your top-k entities, enrich **all of them in one query** — parents, blockers, risks, owners as `ARRAY_AGG(STRUCT(...))` groups seeded by `IN UNNEST(@entity_ids)` — instead of 4×k point lookups.

```python
retriever.enrich_batch(["proj-atlas", "proj-nimbus", "goal-fy26-revenue"])
# → {entity_id: EnrichmentResult(parents=[...], blockers=[...], risks=[...], owners=[...])}
```

## Bring your own ontology

The graph vocabulary is **data, not code**. A pydantic-validated registry declares your entity
types, relationship types, their direction, which ones may recurse, per-type depth caps, and
semantic aliases:

```yaml
# ontology/org_graph.yaml (excerpt)
relationship_types:
  - name: depends_on
    description: The source cannot finish until the target does.
    source_types: [Task, Project]
    target_types: [Task, Project]
    inverse: blocks
    traversal: transitive          # may recurse; `terminal` types are enrichment-only
    canonical_direction: source_to_target
    max_depth: 5                   # per-type blast-radius cap

semantics:
  upstream:  [blocks, depends_on]  # callers ask for a MEANING, not a column value
  hierarchy: [belongs_to]
```

Pattern modules never see a type name. The backend resolves a semantic to a concrete list and
passes it as an array query parameter, so `patterns/` carries no entity- or relationship-type
literals and no table names — those come from the registry's `table_config` too. The one
deliberate exception is documented where it lives: `patterns/blocker_chains.py` orders results
by a `CASE status WHEN 'blocked' ... 'at_risk' ... 'delayed'` priority. That is display
ordering over a status vocabulary the registry does not model, and it degrades to the `ELSE`
branch — not an error — on a domain that uses different status values.

**Adapting to a different domain is two views and one YAML file — no Python changes.** Expose
your own tables in the shape the registry declares. Four node columns and three edge columns
are the whole requirement; anything richer is opt-in through `table_config.node_extra_columns`:

```sql
CREATE VIEW graph_nodes AS
  SELECT id AS node_id, subject AS title, 'Incident' AS node_type, state FROM incidents
  UNION ALL
  SELECT id, name, 'Service', status FROM services;

CREATE VIEW graph_edges AS
  SELECT effect_id AS src_id, cause_id AS dst_id, 'caused_by' AS edge_type FROM incident_links
  UNION ALL
  SELECT service_id, host_id, 'runs_on' FROM service_hosts;
```

```python
from graph_rag.backends.duckdb_backend import DuckDBGraphBackend
from graph_rag.models import GraphFilters
from graph_rag.ontology import FileOntologySource, Ontology, resolve_semantic
from graph_rag.retriever import GraphRetriever

ontology  = Ontology.from_source(FileOntologySource("my_domain.yaml"))
backend   = DuckDBGraphBackend("my_warehouse.duckdb", ontology)   # where the views above live
retriever = GraphRetriever(backend)

retriever.traverse(GraphFilters(
    entity_id="inc-1",
    rel_type=resolve_semantic(ontology, "upstream"),
    rel_max_depth=2,
))
# → [{'entity_id': 'inc-2', 'name': 'Middle Incident', ..., 'depth': 1},
#    {'entity_id': 'inc-3', 'name': 'Leaf Incident',  ..., 'depth': 2}]
```

The `my_domain.yaml` that drives this is [`tests/fixtures/tiny_domain_alt_schema.yaml`](tests/fixtures/tiny_domain_alt_schema.yaml).

This is **verified, not asserted**, along two independent axes.
`tests/test_conformance.py` runs the patterns against a second ontology (Service / Incident /
Team) sharing no type names with the demo graph, *and* runs all eight `GraphBackend` methods
against a third whose table and column names differ from the demo's entirely — asserting both
return identical results. Neither needs a change to any pattern module.

The registry can also live in a **table** rather than a file, behind the same `OntologySource`
Protocol — so adding a traversable relationship type is an `INSERT`, not a redeploy.

## Quickstart

### 2-minute demo — no cloud, no keys

```bash
git clone https://github.com/Gowri-Shankar/graph-rag-sql && cd graph-rag-sql
pip install -e .
python demo.py
```

Loads a bundled synthetic org graph (~375 entities, ~600 relationships, seeded and deterministic) into in-memory DuckDB and prints the blocker chain behind *Project Atlas*, its hierarchy, owners, risks, and a one-query batch enrichment.

```
======================================================================
Graph stats
======================================================================
375 entities, 531 relationships (in-memory DuckDB)

======================================================================
What blocks Project Atlas?
======================================================================
5 blockers found, up to 5 hops deep:

  [distance 1] Atlas blocker task 1 -[blocks]-> Project Atlas
  [distance 2] Atlas blocker task 2 -[blocks]-> Atlas blocker task 1 -[blocks]-> Project Atlas
  [distance 3] Atlas blocker task 3 -[blocks]-> Atlas blocker task 2 -[blocks]-> Atlas blocker task 1 -[blocks]-> Project Atlas
  [distance 4] Atlas blocker task 4 -[blocks]-> Atlas blocker task 3 -[blocks]-> Atlas blocker task 2 -[blocks]-> Atlas blocker task 1 -[blocks]-> Project Atlas
  [distance 5] Atlas blocker task 5 -[blocks]-> Atlas blocker task 4 -[blocks]-> Atlas blocker task 3 -[blocks]-> Atlas blocker task 2 -[blocks]-> Atlas blocker task 1 -[blocks]-> Project Atlas

======================================================================
Project Atlas hierarchy (up to its Goal)
======================================================================
  depth 1: Q1 Resilience Initiative 0 (Initiative)
  depth 2: FY26 Quality Goal 0 (Goal)

======================================================================
Owners and risks
======================================================================
Owners: Rowan Carter
Risks:  Risk: security gap #0

======================================================================
Batch enrichment: 1 query vs the 12+ queries this replaces
======================================================================
Enriched 3 entities with ONE query (naive approach: 3 x 4 lookups = 12+ queries)

  proj-atlas: 2 ancestors, 2 blockers, 1 risks, 1 owners
  proj-1: 2 ancestors, 0 blockers, 0 risks, 1 owners
  proj-0: 2 ancestors, 0 blockers, 0 risks, 1 owners

======================================================================
Goal status summary (recursive counts)
======================================================================
  FY26 Expansion Goal 3 [not_started]: 3 initiatives, 9 projects, 50 tasks
  FY26 Expansion Goal 4 [in_progress]: 3 initiatives, 8 projects, 45 tasks
  FY26 Quality Goal 0 [not_started]: 3 initiatives, 10 projects, 55 tasks
  FY26 Quality Goal 2 [in_progress]: 3 initiatives, 9 projects, 50 tasks
  FY26 Retention Goal 1 [at_risk]: 3 initiatives, 9 projects, 50 tasks

Total runtime: 0.32s
```

### Production path — BigQuery (free sandbox works)

```bash
pip install -e ".[bigquery]"
export GCP_PROJECT_ID=your-gcp-project-id
export BQ_DATASET_ID=graph_rag_demo
python scripts/setup_bigquery.py # creates dataset + tables, loads the CSVs
python demo.py --backend bigquery
```

The [BigQuery sandbox](https://cloud.google.com/bigquery/docs/sandbox) needs only a Google account — no billing.

## Architecture

```mermaid
flowchart LR
 GEN["generator.py<br/>seeded synthetic org graph"] --> CSV["data/*.csv<br/>(&lt; 1 MB, bundled)"]
 CSV -->|"demo.py (default)"| DUCK[("DuckDB<br/>in-memory")]
 CSV -->|"scripts/setup_bigquery.py"| BQ[("BigQuery<br/>sandbox / prod")]
 ONT["ontology/*.yaml or table<br/>OntologySource Protocol"] --> RET
 DIA["dialects/<br/>SqlDialect Protocol"] -.-> PATTERNS
 subgraph PATTERNS["patterns/ — written once, dialect-rendered"]
 P1["1 bounded recursive hierarchy"]
 P2["2 blocker chains + path arrays"]
 P3["3 UNNEST batch enrichment"]
 end
 DUCK --> RET["GraphRetriever<br/>(backends share one Protocol)"]
 BQ --> RET
 PATTERNS -.-> DUCK
 PATTERNS -.-> BQ
 RET --> DEMO["demo.py<br/>'What blocks Project Atlas?'"]
 RET --> BENCH["scripts/benchmark.py<br/>hop latency + bytes billed → cost"]
```

Three seams, all Protocols: `OntologySource` (file or table) supplies the vocabulary, `SqlDialect` absorbs engine differences, and `GraphBackend` is what the patterns run against. `GraphRetriever` is a thin facade mapping a typed `GraphFilters` model onto backend calls — so swapping DuckDB ↔ BigQuery is a constructor argument, and swapping domains is a YAML file.

## Benchmarks

DuckDB numbers below are from `python scripts/benchmark.py --backend duckdb --scale demo --runs 5` on a single run of the demo-scale synthetic graph (375 entities, 531 relationships), on a Windows 11 laptop with an AMD Ryzen 5 4600H and DuckDB 1.5.5 — a relative-shape reference, not a formal benchmark. The BigQuery column wasn't run for this table (no sandbox project provisioned for this session) — run `python scripts/benchmark.py --backend bigquery --scale demo` yourself to reproduce it; the script reports `total_bytes_processed` and an estimated on-demand cost for that column automatically.

| Traversal | DuckDB (local), median / p95 ms | BigQuery |
|---|---|---|
| 1-hop | 11.22 / 11.75 | run `scripts/benchmark.py --backend bigquery` to reproduce |
| 2-hop | 12.20 / 13.52 | run `scripts/benchmark.py --backend bigquery` to reproduce |
| 3-hop | 13.24 / 13.37 | run `scripts/benchmark.py --backend bigquery` to reproduce |
| 5-hop | 16.30 / 16.83 | run `scripts/benchmark.py --backend bigquery` to reproduce |
| `find_blockers` (5-hop) | 26.54 / 27.94 | run `scripts/benchmark.py --backend bigquery` to reproduce |
| `enrich_entities_batch` (10 ids) | 66.51 / 73.22 | run `scripts/benchmark.py --backend bigquery` to reproduce |

Reproduce: `pip install -e ".[bench,bigquery]" && python scripts/benchmark.py`

The headline from the referenced production write-up: retrieval was never the bottleneck — LLM generation dominated end-to-end latency, which is exactly what makes warehouse-speed traversal acceptable for RAG.

## When NOT to do this

Honesty section — this pattern has a clearly bounded sweet spot:

- **Unbounded pathfinding** (shortest path, arbitrary-depth reachability): use a graph database.
- **Frontier explosion**: dense graphs where each hop multiplies rows; the depth bound protects you, but recall suffers.
- **Millisecond OLTP lookups**: warehouse latency floors are tens-to-hundreds of ms.
- **>5 hops or graph algorithms** (PageRank, community detection): wrong tool.
- **Cycle discipline is on you**: bounded depth guarantees termination even with cycles, but cycles inflate results — this repo's generator enforces an acyclic `blocks`/`depends_on` subgraph at write time, which is the correct place to do it.

## Design notes

- **No LangChain / LlamaIndex.** This is the retrieval layer; fewer layers means every query is inspectable. There is deliberately **no LLM dependency** at all.
- **Each query is written once.** Patterns render through a small `SqlDialect` Protocol (~8 members: parameter style, table qualification, array literal/append/membership, struct aggregation, empty-array handling, top-n-per-group) rather than being authored per engine. Adding a backend is one adapter class, not a rewrite of every pattern.
- **No SQL transpiler.** SQLGlot and friends were considered and rejected: recursive CTEs with array operations are exactly where transpilers get unreliable, and generated SQL would undercut the point that every query here is readable.
- **Parameterized by design:** relationship types are always passed as array query parameters (`IN UNNEST(@rel_types)`), never string-interpolated into SQL, with a regression test asserting parameterization.
- **Deterministic by construction:** the synthetic graph generates from a fixed seed with timestamps derived from a fixed base date — running the generator twice yields byte-identical CSVs.

## Roadmap

Designed but **not built** — listed with the seam each one drops into, so the claim is checkable:

- **Postgres dialect** — one `SqlDialect` implementation in `src/graph_rag/dialects/postgres.py`.
  `WITH RECURSIVE` is standard SQL; the engine differences are already isolated behind the
  Protocol (Postgres notably lacks `QUALIFY`, which is why `top_n_per_group` is a dialect member).
- **MCP server** — tool schemas generated from the registry, which is why `description` is a
  required field on every entity and relationship type.
- **Real public datasets** — loaders for deps.dev, MITRE ATT&CK, or GLEIF ownership. The registry
  is what makes this a data task rather than a rewrite.

## Provenance & license

Explores an idea described publicly in the engineering blog post linked above. All entities, people, and organizations in the bundled data are fictional. Licensed under MIT — see [LICENSE](LICENSE).
