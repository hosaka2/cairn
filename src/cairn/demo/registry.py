"""Local registration of the demo components.

`CAIRN_REGISTRY=cairn.demo.registry` (the default) makes the web app and CLI import this,
registering the Ingestor, the Evaluators and the inference unit (Runner + process_one +
Config) so that adding data and running evals from the UI goes through the demo scripts.
"""

from __future__ import annotations

from cairn.demo.evaluate import AnomalyEvalV1, AnomalyEvalV2
from cairn.demo.ingest import SyntheticIngest
from cairn.demo.model import ChunkRunner, ModelConfig, make_process_one
from cairn.registry import register_evaluator, register_inference, register_ingestor

# Both synthetic datasets are ingested by SyntheticIngest.
for _ds in ("sensor-anomaly-A", "sensor-anomaly-B"):
    register_ingestor(_ds)(SyntheticIngest)

# Per eval table: the v1/v2 Evaluators and the inference unit (Runner + process_one + Config).
for _t in ("anomaly", "anomaly-siteB"):
    register_evaluator(_t, "v1")(AnomalyEvalV1)
    register_evaluator(_t, "v2")(AnomalyEvalV2)
    register_inference(_t, runner=ChunkRunner, process_factory=make_process_one, config=ModelConfig)
