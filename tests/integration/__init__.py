"""tests.integration package

This sub-package groups **integration-level** test suites that exercise the
HeronAI application via its public interfaces (HTTP endpoints, CLI wrappers).
Keeping integration tests separate from *unit* tests allows developers to run
only the fast unit subset during TDD cycles while still enabling full-stack
validation in CI via `pytest -m integration`.
"""
