"""cairn — an append-only registry for evaluations and datasets.

The single source of truth is object storage; there is no database server.
Everything public goes through `cairn.core`, which the CLI and the web UI both call.
"""

from cairn.core.config import Config, load_config
from cairn.core.storage import Storage

__version__ = "0.0.1"
__all__ = ["Config", "Storage", "__version__", "load_config"]
