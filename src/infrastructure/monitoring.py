import os
from typing import Any

import psutil

from src.core.interfaces import SystemMonitorInterface


class PsutilSystemMonitor(SystemMonitorInterface):
    """Concrete implementation for system monitoring"""

    def __init__(self):
        self.process = psutil.Process(os.getpid())

    def get_process_memory_mb(self) -> float:
        """Get memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024

    def get_cpu_percent(self) -> float:
        """Get CPU usage in %"""
        return psutil.cpu_percent()

    def get_memory_info(self) -> Any:
        """Get memory info"""
        return self.process.memory_info()
