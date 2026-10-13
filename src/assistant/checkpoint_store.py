
import atexit
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DB = PROJECT_ROOT / "data" / "database" / "chat_checkpoints.sqlite"

CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)

# Keep one connection alive for the lifetime of the application.
_context = SqliteSaver.from_conn_string(str(CHECKPOINT_DB))
checkpointer = _context.__enter__()

atexit.register(_context.__exit__, None, None, None)


def get_checkpointer() -> SqliteSaver:
    """Return the shared SQLite checkpointer."""
    return checkpointer
