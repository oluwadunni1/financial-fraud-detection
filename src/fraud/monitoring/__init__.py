"""Label-free monitoring: delayed labels, performance estimation, drift.

Runs in `.venv-monitoring`, not `.venv`: NannyML's pins conflict with the main
environment (see pyproject.toml). Nothing in this package may import xgboost or
torch -- monitoring reads scores the models already wrote, it never loads a
model -- and tests/test_monitoring.py enforces that.
"""
