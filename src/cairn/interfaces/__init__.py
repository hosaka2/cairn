"""The interfaces a script implements.

Ingestor  : turns any input into rows that conform to schema.yaml
Runner    : decides how inference work is split into units, without running it
Evaluator : turns predictions and dataset rows into an EvalResult
"""

from cairn.interfaces.evaluator import DatasetView, EvalContext, Evaluator, PredictionView
from cairn.interfaces.ingestor import IngestContext, Ingestor
from cairn.interfaces.runner import Runner

__all__ = [
    "DatasetView",
    "EvalContext",
    "Evaluator",
    "IngestContext",
    "Ingestor",
    "PredictionView",
    "Runner",
]
