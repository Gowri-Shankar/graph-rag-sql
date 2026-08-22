"""The three traversal patterns, each written once against the `SqlDialect` Protocol.

No pattern module names a table, a column, or a relationship/entity type. Table and column
names come from `Ontology.table_config`; relationship types arrive pre-resolved (see
`graph_rag.ontology.resolve.resolve_semantic`) as plain parameter values. A backend is free to
swap dialects, ontologies, or resolved type lists without ever touching this package.
"""
