"""Seeded synthetic org-graph generator for the fictional company "Acme Analytics".

This generator is purpose-built for the org scenario (goals -> initiatives -> projects ->
tasks, owned by people, threatened by risks) rather than driven generically off the ontology
registry — a generic driver would produce a uniform, boring graph and fight the deliberate
"Project Atlas" blocker chain planted below. Its OUTPUT is validated against the registry via
`graph_rag.ontology.resolve.validate_edges`, so the data and the vocabulary can never drift
apart even though the generator itself is hand-authored.

Determinism: every random choice comes from a `random.Random(seed)` instance, and every
timestamp is derived from `seed` and a fixed base date — never `datetime.now()`. Running this
generator twice with the same seed produces byte-identical CSVs.

Hierarchy direction: `belongs_to` edges point from child to parent (Task -> Project ->
Initiative -> Goal), i.e. `source_entity_id` is the child and `target_entity_id` is the
parent. This direction matters for the traversal patterns ported in a later milestone.

Acyclicity: `blocks` and `depends_on` edges always point from an earlier node to a later node
in a fixed topological ordering assigned during generation (source's order index is always
less than the target's). A transitive traversal has no depth bound baked into the *data*, only
into the *query* (see the ontology's `max_depth` caps) — without the topological ordering, a
cyclic graph would make an unbounded recursive CTE never terminate, and even a depth-bounded
one would return duplicate, inflated paths. Guaranteeing acyclicity at write time avoids both.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from graph_rag.models import Entity, Relationship

BASE_DATE = datetime(2025, 1, 1)  # noqa: DTZ001 — naive by design, synthetic data only

ORG_UNITS = ["Platform", "Growth", "Data", "EMEA"]

STATUSES_TASK = ["not_started", "in_progress", "completed", "blocked", "at_risk"]
STATUSES_PROJECT = ["not_started", "in_progress", "completed", "at_risk", "blocked"]
PRIORITIES = ["critical", "high", "medium", "low"]
RISK_LEVELS = ["low", "medium", "high", "critical"]

FIRST_NAMES = [
    "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Avery", "Quinn", "Rowan",
    "Skyler", "Reese", "Emerson", "Finley", "Hayden", "Elliot", "Cameron", "Drew",
    "Blake", "Sawyer", "Sage", "Parker", "Dakota", "Kendall", "Peyton", "Marlowe",
    "Harper", "Jules", "Remy", "Shay", "Tatum", "Wren", "Ari", "Bex", "Cy",
    "Denny", "Frankie", "Gale", "Hollis", "Ira", "Jules", "Kip",
]
LAST_NAMES = [
    "Bennett", "Carter", "Delgado", "Ellison", "Foster", "Grant", "Holloway",
    "Ingram", "Jansen", "Keller", "Lindqvist", "Mercer", "Nakamura", "Osei",
    "Pierce", "Quaid", "Reyes", "Sato", "Tran", "Ueda", "Vance", "Wexler",
    "Xu", "Yates", "Zabala", "Abara", "Brandt", "Cortez", "Duvall", "Estrada",
    "Farrow", "Gable", "Huxley", "Ionescu", "Jovanovic", "Kowalski", "Lund",
    "Mbeki", "Novak", "Okafor",
]

FICTIONAL_INITIATIVE_ADJECTIVES = ["Growth", "Modernization", "Expansion", "Resilience", "Velocity"]
FICTIONAL_PROJECT_VERBS = ["Migrate", "Rebuild", "Launch", "Automate", "Consolidate", "Refactor"]
FICTIONAL_PROJECT_NOUNS = [
    "auth service", "billing pipeline", "onboarding flow", "search index",
    "reporting dashboard", "data warehouse", "notification system", "checkout flow",
    "partner API", "mobile app", "support portal", "analytics events",
]
FICTIONAL_TASK_VERBS = ["Write", "Review", "Deploy", "Test", "Design", "Document", "Fix", "Update"]
FICTIONAL_TASK_NOUNS = [
    "schema migration", "API contract", "load test", "rollback plan", "config",
    "integration test", "runbook", "dashboard panel", "alert rule", "index",
]

ATLAS_PROJECT_NAME = "Project Atlas"
ATLAS_CHAIN_DEPTH = 5


@dataclass
class GraphBuilder:
    """Accumulates entities and relationships during generation."""

    rng: random.Random
    entities: list[Entity] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    order_index: dict[str, int] = field(default_factory=dict)
    _next_order: int = 0

    def add_entity(self, entity: Entity, *, orderable: bool = False) -> Entity:
        self.entities.append(entity)
        if orderable:
            self.order_index[entity.entity_id] = self._next_order
            self._next_order += 1
        return entity

    def add_relationship(self, relationship: Relationship) -> None:
        self.relationships.append(relationship)


def _ts(seed: int, offset_minutes: int) -> datetime:
    """Derive a deterministic timestamp from the seed and an offset — never datetime.now()."""
    return BASE_DATE + timedelta(minutes=seed % 1000) + timedelta(minutes=offset_minutes)


def _person_name(rng: random.Random) -> tuple[str, str]:
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    return first, last


def _generate_people(builder: GraphBuilder, seed: int, count: int) -> list[Entity]:
    people = []
    for i in range(count):
        first, last = _person_name(builder.rng)
        email = f"{first.lower()}.{last.lower()}{i}@example.com"
        entity = Entity(
            entity_id=f"person-{i}",
            name=f"{first} {last}",
            type="Person",
            status="active",
            owner_id=None,
            description=None,
            priority=None,
            risk_level=None,
            properties={"email": email, "org_unit": builder.rng.choice(ORG_UNITS)},
            created_at=_ts(seed, i),
            updated_at=None,
        )
        builder.add_entity(entity)
        people.append(entity)
    return people


def _generate_risks(builder: GraphBuilder, seed: int, count: int) -> list[Entity]:
    risks = []
    for i in range(count):
        entity = Entity(
            entity_id=f"risk-{i}",
            name=f"Risk: {builder.rng.choice(['vendor delay', 'staffing gap', 'scope creep', 'security gap', 'budget cut'])} #{i}",
            type="Risk",
            status="open",
            owner_id=None,
            description="Identified risk tracked against one or more initiatives or projects.",
            priority=builder.rng.choice(PRIORITIES),
            risk_level=builder.rng.choice(RISK_LEVELS),
            properties={},
            created_at=_ts(seed, 1000 + i),
            updated_at=None,
        )
        builder.add_entity(entity)
        risks.append(entity)
    return risks


def _email_for(person: Entity) -> str:
    return person.properties["email"]


def _generate_goals(builder: GraphBuilder, seed: int, count: int, people: list[Entity]) -> list[Entity]:
    goals = []
    for i in range(count):
        owner = builder.rng.choice(people)
        entity = Entity(
            entity_id=f"goal-{i}",
            name=f"FY26 {builder.rng.choice(['Revenue', 'Retention', 'Efficiency', 'Expansion', 'Quality'])} Goal {i}",
            type="Goal",
            status=builder.rng.choice(["not_started", "in_progress", "at_risk"]),
            owner_id=_email_for(owner),
            description="A top-level company objective for the fiscal year.",
            priority="critical",
            risk_level=builder.rng.choice(RISK_LEVELS),
            properties={},
            created_at=_ts(seed, 2000 + i),
            updated_at=None,
        )
        builder.add_entity(entity)
        goals.append(entity)
        builder.add_relationship(
            Relationship(
                source_entity_id=owner.entity_id,
                target_entity_id=entity.entity_id,
                relationship_type="accountable_for",
                confidence=1.0,
                created_at=_ts(seed, 2000 + i),
            )
        )
    return goals


def _generate_initiatives(
    builder: GraphBuilder, seed: int, count: int, goals: list[Entity], people: list[Entity]
) -> list[Entity]:
    initiatives = []
    for i in range(count):
        goal = goals[i % len(goals)]
        owner = builder.rng.choice(people)
        adjective = builder.rng.choice(FICTIONAL_INITIATIVE_ADJECTIVES)
        entity = Entity(
            entity_id=f"init-{i}",
            name=f"Q{builder.rng.randint(1, 4)} {adjective} Initiative {i}",
            type="Initiative",
            status=builder.rng.choice(["not_started", "in_progress", "at_risk", "completed"]),
            owner_id=_email_for(owner),
            description="A cross-team program of work that advances a company goal.",
            priority=builder.rng.choice(PRIORITIES),
            risk_level=builder.rng.choice(RISK_LEVELS),
            properties={"org_unit": builder.rng.choice(ORG_UNITS)},
            created_at=_ts(seed, 3000 + i),
            updated_at=None,
        )
        builder.add_entity(entity)
        initiatives.append(entity)
        builder.add_relationship(
            Relationship(
                source_entity_id=entity.entity_id,
                target_entity_id=goal.entity_id,
                relationship_type="belongs_to",
                confidence=1.0,
                created_at=_ts(seed, 3000 + i),
            )
        )
        builder.add_relationship(
            Relationship(
                source_entity_id=owner.entity_id,
                target_entity_id=entity.entity_id,
                relationship_type="owns",
                confidence=1.0,
                created_at=_ts(seed, 3000 + i),
            )
        )
    return initiatives


def _generate_projects(
    builder: GraphBuilder, seed: int, count: int, initiatives: list[Entity], people: list[Entity]
) -> list[Entity]:
    projects = []
    for i in range(count):
        initiative = initiatives[i % len(initiatives)]
        owner = builder.rng.choice(people)
        verb = builder.rng.choice(FICTIONAL_PROJECT_VERBS)
        noun = builder.rng.choice(FICTIONAL_PROJECT_NOUNS)
        entity = Entity(
            entity_id=f"proj-{i}",
            name=f"{verb} {noun} {i}",
            type="Project",
            status=builder.rng.choice(STATUSES_PROJECT),
            owner_id=_email_for(owner),
            description="A scoped body of work delivering a specific outcome within an initiative.",
            priority=builder.rng.choice(PRIORITIES),
            risk_level=builder.rng.choice(RISK_LEVELS),
            properties={"org_unit": builder.rng.choice(ORG_UNITS)},
            created_at=_ts(seed, 4000 + i),
            updated_at=None,
        )
        builder.add_entity(entity, orderable=True)
        projects.append(entity)
        builder.add_relationship(
            Relationship(
                source_entity_id=entity.entity_id,
                target_entity_id=initiative.entity_id,
                relationship_type="belongs_to",
                confidence=1.0,
                created_at=_ts(seed, 4000 + i),
            )
        )
        builder.add_relationship(
            Relationship(
                source_entity_id=owner.entity_id,
                target_entity_id=entity.entity_id,
                relationship_type="owns",
                confidence=1.0,
                created_at=_ts(seed, 4000 + i),
            )
        )
    return projects


def _generate_tasks(
    builder: GraphBuilder, seed: int, count: int, projects: list[Entity], people: list[Entity]
) -> list[Entity]:
    tasks = []
    for i in range(count):
        project = projects[i % len(projects)]
        verb = builder.rng.choice(FICTIONAL_TASK_VERBS)
        noun = builder.rng.choice(FICTIONAL_TASK_NOUNS)
        entity = Entity(
            entity_id=f"task-{i}",
            name=f"{verb} {noun} {i}",
            type="Task",
            status=builder.rng.choice(STATUSES_TASK),
            owner_id=_email_for(builder.rng.choice(people)),
            description="A unit of work within a project.",
            priority=builder.rng.choice(PRIORITIES),
            risk_level=None,
            properties={},
            created_at=_ts(seed, 5000 + i),
            updated_at=None,
        )
        builder.add_entity(entity, orderable=True)
        tasks.append(entity)
        builder.add_relationship(
            Relationship(
                source_entity_id=entity.entity_id,
                target_entity_id=project.entity_id,
                relationship_type="belongs_to",
                confidence=1.0,
                created_at=_ts(seed, 5000 + i),
            )
        )
    return tasks


def _generate_risk_edges(builder: GraphBuilder, seed: int, risks: list[Entity], targets: list[Entity]) -> None:
    for i, risk in enumerate(risks):
        target = builder.rng.choice(targets)
        builder.add_relationship(
            Relationship(
                source_entity_id=risk.entity_id,
                target_entity_id=target.entity_id,
                relationship_type="threatens",
                confidence=round(builder.rng.uniform(0.5, 1.0), 2),
                created_at=_ts(seed, 6000 + i),
            )
        )


def _generate_blocking_edges(
    builder: GraphBuilder, seed: int, orderable: list[Entity], edge_count: int
) -> None:
    """Add random blocks/depends_on edges, always from a lower order-index node to a higher one.

    Both relationship types point "upstream to downstream" here: source is upstream of (must
    resolve before) target. Only allowing source_index < target_index guarantees the resulting
    subgraph is acyclic, since no edge can ever point backwards in that fixed ordering.
    """
    attempts = 0
    added = 0
    while added < edge_count and attempts < edge_count * 20:
        attempts += 1
        a, b = builder.rng.sample(orderable, 2)
        if builder.order_index[a.entity_id] >= builder.order_index[b.entity_id]:
            a, b = b, a
        if builder.order_index[a.entity_id] >= builder.order_index[b.entity_id]:
            continue
        rel_type = builder.rng.choice(["blocks", "depends_on"])
        builder.add_relationship(
            Relationship(
                source_entity_id=a.entity_id,
                target_entity_id=b.entity_id,
                relationship_type=rel_type,
                confidence=round(builder.rng.uniform(0.6, 1.0), 2),
                created_at=_ts(seed, 7000 + added),
            )
        )
        added += 1


def _plant_atlas(
    builder: GraphBuilder,
    seed: int,
    projects: list[Entity],
    people: list[Entity],
    risks: list[Entity],
    initiative: Entity,
) -> None:
    """Plant the "Project Atlas" demo anchor: a named project with a >=5-hop blocker chain.

    The chain is a fresh, dedicated set of tasks that participate in no other blocks/depends_on
    edges, so it is trivially acyclic and cannot interact with the randomly generated subgraph's
    acyclicity guarantee.
    """
    atlas = Entity(
        entity_id="proj-atlas",
        name=ATLAS_PROJECT_NAME,
        type="Project",
        status="at_risk",
        owner_id=_email_for(people[0]),
        description="Flagship platform migration project used as the repo's demo anchor.",
        priority="critical",
        risk_level="high",
        properties={"org_unit": "Platform"},
        created_at=_ts(seed, 8000),
        updated_at=None,
    )
    builder.add_entity(atlas, orderable=True)
    projects.append(atlas)

    builder.add_relationship(
        Relationship(
            source_entity_id=atlas.entity_id,
            target_entity_id=initiative.entity_id,
            relationship_type="belongs_to",
            confidence=1.0,
            created_at=_ts(seed, 8000),
        )
    )
    builder.add_relationship(
        Relationship(
            source_entity_id=people[0].entity_id,
            target_entity_id=atlas.entity_id,
            relationship_type="owns",
            confidence=1.0,
            created_at=_ts(seed, 8000),
        )
    )
    builder.add_relationship(
        Relationship(
            source_entity_id=risks[0].entity_id,
            target_entity_id=atlas.entity_id,
            relationship_type="threatens",
            confidence=0.9,
            created_at=_ts(seed, 8001),
        )
    )

    chain_target_id = atlas.entity_id
    for depth in range(1, ATLAS_CHAIN_DEPTH + 1):
        blocker = Entity(
            entity_id=f"task-atlas-blocker-{depth}",
            name=f"Atlas blocker task {depth}",
            type="Task",
            status="blocked" if depth % 2 == 0 else "at_risk",
            owner_id=_email_for(people[depth % len(people)]),
            description="Task in the planted blocker chain behind Project Atlas.",
            priority="critical",
            risk_level=None,
            properties={},
            created_at=_ts(seed, 8010 + depth),
            updated_at=None,
        )
        builder.add_entity(blocker, orderable=True)
        builder.add_relationship(
            Relationship(
                source_entity_id=blocker.entity_id,
                target_entity_id=atlas.entity_id,
                relationship_type="belongs_to",
                confidence=1.0,
                created_at=_ts(seed, 8010 + depth),
            )
        )
        builder.add_relationship(
            Relationship(
                source_entity_id=blocker.entity_id,
                target_entity_id=chain_target_id,
                relationship_type="blocks",
                confidence=1.0,
                created_at=_ts(seed, 8010 + depth),
            )
        )
        chain_target_id = blocker.entity_id


def generate_org_graph(
    seed: int = 42, scale: str = "demo"
) -> tuple[list[Entity], list[Relationship]]:
    """Generate a deterministic, ontology-conformant synthetic org graph.

    Args:
        seed: Random seed; identical seeds produce byte-identical output.
        scale: ``"demo"`` (~375 nodes, ~600 edges, bundled in git) or ``"large"``
            (~20k nodes, ~60k edges, for benchmarking — not bundled).

    Returns:
        A tuple of (entities, relationships).
    """
    if scale == "demo":
        n_goals, n_initiatives, n_projects, n_tasks, n_people, n_risks = 5, 15, 44, 245, 40, 20
        n_blocking_edges = 130
    elif scale == "large":
        n_goals, n_initiatives, n_projects, n_tasks, n_people, n_risks = 20, 200, 2000, 17000, 600, 400
        n_blocking_edges = 8000
    else:
        raise ValueError(f"unknown scale: {scale!r}")

    rng = random.Random(seed)
    builder = GraphBuilder(rng=rng)

    people = _generate_people(builder, seed, n_people)
    risks = _generate_risks(builder, seed, n_risks)
    goals = _generate_goals(builder, seed, n_goals, people)
    initiatives = _generate_initiatives(builder, seed, n_initiatives, goals, people)
    projects = _generate_projects(builder, seed, n_projects, initiatives, people)
    tasks = _generate_tasks(builder, seed, n_tasks, projects, people)

    _plant_atlas(builder, seed, projects, people, risks, initiatives[0])

    orderable = [
        e
        for e in builder.entities
        if e.entity_id in builder.order_index and not e.entity_id.startswith("proj-atlas") and not e.entity_id.startswith("task-atlas-blocker-")
    ]
    _generate_blocking_edges(builder, seed, orderable, n_blocking_edges)
    _generate_risk_edges(builder, seed, risks, goals + initiatives + projects + tasks)

    return builder.entities, builder.relationships


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_csvs(
    entities: list[Entity], relationships: list[Relationship], out_dir: Path
) -> None:
    """Write entities and relationships to `entities.csv` and `relationships.csv`."""
    entity_fields = [
        "entity_id", "name", "type", "status", "owner_id", "description",
        "priority", "risk_level", "properties", "created_at", "updated_at",
    ]
    entity_rows = [
        {
            "entity_id": e.entity_id,
            "name": e.name,
            "type": e.type,
            "status": e.status or "",
            "owner_id": e.owner_id or "",
            "description": e.description or "",
            "priority": e.priority or "",
            "risk_level": e.risk_level or "",
            "properties": json.dumps(e.properties, sort_keys=True),
            "created_at": e.created_at.isoformat(),
            "updated_at": e.updated_at.isoformat() if e.updated_at else "",
        }
        for e in entities
    ]
    _write_csv(out_dir / "entities.csv", entity_rows, entity_fields)

    rel_fields = [
        "source_entity_id", "target_entity_id", "relationship_type", "confidence", "created_at",
    ]
    rel_rows = [
        {
            "source_entity_id": r.source_entity_id,
            "target_entity_id": r.target_entity_id,
            "relationship_type": r.relationship_type,
            "confidence": r.confidence,
            "created_at": r.created_at.isoformat(),
        }
        for r in relationships
    ]
    _write_csv(out_dir / "relationships.csv", rel_rows, rel_fields)


def main() -> None:
    """CLI entry point: `python -m graph_rag.generator [--scale demo|large]`."""
    parser = argparse.ArgumentParser(description="Generate the synthetic org graph.")
    parser.add_argument("--scale", choices=["demo", "large"], default="demo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    entities, relationships = generate_org_graph(seed=args.seed, scale=args.scale)
    write_csvs(entities, relationships, args.out_dir)
    print(f"Wrote {len(entities)} entities and {len(relationships)} relationships to {args.out_dir}")


if __name__ == "__main__":
    main()
