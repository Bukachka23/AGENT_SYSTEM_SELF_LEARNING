import gc
import tracemalloc
from contextlib import contextmanager
from typing import Any, Generator

from src.config.logger import Logger

logger = Logger()

@contextmanager
def memory_monitor(log: Logger, task_name) -> Generator[None, Any, None]:
    """Monitor memory usage during a task."""
    tracemalloc.start()
    snapshot_start = tracemalloc.take_snapshot()

    yield

    snapshot_end = tracemalloc.take_snapshot()
    top_stats = snapshot_end.compare_to(snapshot_start, "lineno")
    log.info(f"\nMemory usage for {task_name}:")
    for stat in top_stats[:13]:
        log.info(stat)

    tracemalloc.stop()
    gc.collect()
