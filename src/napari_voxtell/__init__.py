import os

# Reduce CUDA memory fragmentation for large multi-prompt runs. Best-effort: only
# takes effect if set before torch initialises its CUDA caching allocator, so we set
# it before importing the widget (which imports torch).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

__version__ = "0.1.0"
from .widget_main import VoxtellWidget

__all__ = ("VoxtellWidget",)
