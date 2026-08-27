"""Where inference actually runs.

base  : the OrchestratorAdapter protocol
local : runs the work in this process, which is what the CLI and web UI use
"""

from cairn.adapters.base import OrchestratorAdapter
from cairn.adapters.local import InlineAdapter

__all__ = ["InlineAdapter", "OrchestratorAdapter"]
