"""Core of cairn: everything the platform fixes, as opposed to user scripts.

storage : object storage access through fsspec
config  : CAIRN_ROOT and other runtime settings
ids     : ULID generation
schema  : parsing and validation of schema.yaml / table.yaml
records : types passed between the platform and scripts (Metric, EvalResult, …)
dataset : append, merge and snapshot of datasets
evals   : creating runs, taking predictions, scoring, listing
"""
