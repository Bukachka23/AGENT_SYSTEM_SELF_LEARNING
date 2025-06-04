import gc
import os
from collections import deque
from typing import Any, List

import psutil


class MemoryEfficientStorage:
    """Storage class that automatically manages memory usage with logging."""

    def __init__(self, max_items: int = 1000, memory_threshold: float = 0.8, logger=None):
        self.data = deque(maxlen=max_items)
        self.memory_threshold = memory_threshold
        self._cleanup_counter = 0
        self._cleanup_frequency = 10
        self.logger = logger

        # Statistics
        self.stats = {
            'items_added': 0,
            'items_removed': 0,
            'cleanups_performed': 0,
            'max_memory_reached': 0
        }

        if self.logger:
            self.logger.debug(
                f"MemoryEfficientStorage initialized - "
                f"Max items: {max_items}, Memory threshold: {memory_threshold * 100}%"
            )

    def add(self, item: Any) -> None:
        """Add item with periodic memory checks."""
        self.data.append(item)
        self.stats['items_added'] += 1
        self._cleanup_counter += 1

        if self._cleanup_counter >= self._cleanup_frequency:
            self._cleanup_counter = 0
            current_usage = self._check_memory_usage()

            if self.logger:
                self.logger.debug(f"Memory check - Usage: {current_usage * 100:.1f}%")

            if current_usage > self.memory_threshold:
                if self.logger:
                    self.logger.warning(
                        f"⚠️ Memory threshold exceeded: {current_usage * 100:.1f}% > {self.memory_threshold * 100}%"
                    )
                self.stats['max_memory_reached'] += 1
                self._cleanup_old_items()

    def _check_memory_usage(self) -> float:
        """Check current memory usage percentage."""
        try:
            process = psutil.Process(os.getpid())
            # Get system memory info
            virtual_memory = psutil.virtual_memory()
            process_memory = process.memory_info().rss

            # Calculate percentage of system memory used by this process
            usage = process_memory / virtual_memory.total

            if self.logger and usage > 0.5:
                self.logger.warning(
                    f"High memory usage detected - "
                    f"Process: {process_memory / 1024 / 1024:.1f} MB, "
                    f"System: {usage * 100:.1f}%"
                )

            return usage
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error checking memory usage: {str(e)}")
            return 0.0

    def _cleanup_old_items(self) -> None:
        """Remove oldest items to free memory."""
        initial_size = len(self.data)
        items_to_remove = len(self.data) // 4

        for _ in range(items_to_remove):
            if self.data:
                self.data.popleft()
                self.stats['items_removed'] += 1

        self.stats['cleanups_performed'] += 1

        # Force garbage collection
        collected = gc.collect()

        if self.logger:
            process = psutil.Process(os.getpid())
            current_memory = process.memory_info().rss / 1024 / 1024

            self.logger.info(
                f"🧹 Memory cleanup performed - "
                f"Items: {initial_size} → {len(self.data)} (-{items_to_remove}), "
                f"GC collected: {collected}, "
                f"Memory: {current_memory:.1f} MB"
            )

    def get_recent(self, n: int = 10) -> List[Any]:
        """Get n most recent items."""
        items = list(self.data)[-n:]

        if self.logger:
            self.logger.debug(f"Retrieved {len(items)} recent items")

        return items

    def clear(self) -> None:
        """Clear all data."""
        initial_size = len(self.data)
        self.data.clear()
        collected = gc.collect()

        if self.logger:
            self.logger.info(
                f"🗑️ Storage cleared - "
                f"Items removed: {initial_size}, "
                f"GC collected: {collected} objects"
            )

            # Log final statistics
            self.logger.debug(
                f"Storage statistics - "
                f"Total added: {self.stats['items_added']}, "
                f"Total removed: {self.stats['items_removed']}, "
                f"Cleanups: {self.stats['cleanups_performed']}, "
                f"Memory peaks: {self.stats['max_memory_reached']}"
            )

    def get_stats(self) -> dict:
        """Get storage statistics."""
        return {**self.stats, 'current_size': len(self.data), 'max_size': self.data.maxlen}
