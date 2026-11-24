"""Research paper assistant (modular monolith).

Layers (dependencies point downwards only):
    presentation/   controllers: HTTP API, web pages, CLI, worker job entry points
    services/       business logic and workflows
    repositories/   database access (SQLAlchemy)
    integrations/   external systems: LLM providers, academic APIs, storage
    domain/         shared schemas and errors used by every layer
"""
