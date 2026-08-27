"""Bundled demo: a real pipeline that exercises every interface (synthetic data, no external deps).

`cairn demo-seed` runs it, driving Ingestor -> Runner -> OrchestratorAdapter(InlineAdapter)
-> Evaluator for real rather than faked, producing eval results with actual metrics.
"""

from cairn.demo.pipeline import seed

__all__ = ["seed"]
