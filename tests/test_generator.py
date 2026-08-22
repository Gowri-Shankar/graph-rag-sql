"""Tests for the synthetic org-graph generator."""

from __future__ import annotations

from collections import defaultdict, deque

from graph_rag.generator import ATLAS_CHAIN_DEPTH, ATLAS_PROJECT_NAME, generate_org_graph
from graph_rag.ontology import FileOntologySource, Ontology, validate_edges

ORG_GRAPH_PATH = "ontology/org_graph.yaml"


def test_generation_is_deterministic():
    entities_a, relationships_a = generate_org_graph(seed=42)
    entities_b, relationships_b = generate_org_graph(seed=42)

    assert [e.model_dump() for e in entities_a] == [e.model_dump() for e in entities_b]
    assert [r.model_dump() for r in relationships_a] == [r.model_dump() for r in relationships_b]


def test_blocks_and_depends_on_subgraph_is_acyclic():
    _, relationships = generate_org_graph(seed=42)
    edges = [
        (r.source_entity_id, r.target_entity_id)
        for r in relationships
        if r.relationship_type in ("blocks", "depends_on")
    ]

    adjacency: dict[str, list[str]] = defaultdict(list)
    in_degree: dict[str, int] = defaultdict(int)
    nodes = set()
    for src, dst in edges:
        adjacency[src].append(dst)
        in_degree[dst] += 1
        nodes.add(src)
        nodes.add(dst)

    queue = deque(n for n in nodes if in_degree[n] == 0)
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for neighbor in adjacency[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    assert visited == len(nodes), "blocks/depends_on subgraph contains a cycle"


def test_project_atlas_has_a_deep_blocker_chain():
    entities, relationships = generate_org_graph(seed=42)
    entities_by_id = {e.entity_id: e for e in entities}

    atlas = next(e for e in entities if e.name == ATLAS_PROJECT_NAME)

    blockers_of: dict[str, list[str]] = defaultdict(list)
    for r in relationships:
        if r.relationship_type in ("blocks", "depends_on"):
            blockers_of[r.target_entity_id].append(r.source_entity_id)

    max_depth = 0
    frontier = [(atlas.entity_id, 0)]
    seen = {atlas.entity_id}
    while frontier:
        node, depth = frontier.pop()
        max_depth = max(max_depth, depth)
        for source in blockers_of.get(node, []):
            if source not in seen:
                seen.add(source)
                frontier.append((source, depth + 1))

    assert max_depth >= ATLAS_CHAIN_DEPTH
    assert atlas.status in ("at_risk", "blocked")
    assert atlas.owner_id
    assert entities_by_id


def test_every_task_has_a_belongs_to_path_to_a_goal():
    entities, relationships = generate_org_graph(seed=42)
    entities_by_id = {e.entity_id: e for e in entities}

    parent_of: dict[str, str] = {
        r.source_entity_id: r.target_entity_id
        for r in relationships
        if r.relationship_type == "belongs_to"
    }

    tasks = [e for e in entities if e.type == "Task"]
    assert tasks

    for task in tasks:
        node_id = task.entity_id
        hops = 0
        while entities_by_id[node_id].type != "Goal":
            assert node_id in parent_of, f"{node_id} has no belongs_to parent"
            node_id = parent_of[node_id]
            hops += 1
            assert hops < 20, f"belongs_to chain from {task.entity_id} did not terminate"
        assert entities_by_id[node_id].type == "Goal"


def test_generated_graph_conforms_to_ontology():
    entities, relationships = generate_org_graph(seed=42)
    ontology = Ontology.from_source(FileOntologySource(ORG_GRAPH_PATH))
    validate_edges(ontology, entities, relationships)
