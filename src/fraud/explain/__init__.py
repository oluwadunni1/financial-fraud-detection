"""Why a model scored a transaction the way it did.

`tree` explains the XGBoost champion with exact TreeSHAP and folds encoded
columns back into the fields an analyst recognises. `graph` explains GraphSAGE
with exact Shapley values over groups of its request subgraph. Runs in `.venv`
(the boosters need XGBoost 3.4; .venv-monitoring pins 2.1).
"""
