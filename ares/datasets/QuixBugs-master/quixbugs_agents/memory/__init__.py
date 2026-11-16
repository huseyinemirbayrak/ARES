from pathlib import Path
from ..config import MEMORY_STORAGE_DIR

from .mem0_engine import NativeMem0Memory, Mem0Manager
from .cognee_engine import NativeCogneeEngine, CogneeManager, CogneeNode, CogneeEdge
from .zep_engine import NativeZepTemporalMemory, ZepManager
from .qdrant_engine import NativeQdrantEngine, QdrantManager

def clear_all_memories():
    """Wipes all persisted memory storage files and resets engine singleton instances."""
    Mem0Manager._instance = None
    CogneeManager._instance = None
    ZepManager._instance = None
    QdrantManager._instance = None

    if MEMORY_STORAGE_DIR.exists():
        for f in MEMORY_STORAGE_DIR.glob("*"):
            if f.is_file():
                try:
                    f.unlink()
                except Exception:
                    pass

__all__ = [
    "NativeMem0Memory", "Mem0Manager",
    "NativeCogneeEngine", "CogneeManager", "CogneeNode", "CogneeEdge",
    "NativeZepTemporalMemory", "ZepManager",
    "NativeQdrantEngine", "QdrantManager",
    "clear_all_memories"
]

