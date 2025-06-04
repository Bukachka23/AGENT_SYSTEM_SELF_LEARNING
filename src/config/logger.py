import logging
import sys
from pathlib import Path
from datetime import datetime
import json
from typing import Any, Dict

from src.core.interfaces import LoggerInterface


class Logger(LoggerInterface):
    """Enhanced logger with structured logging and performance tracking."""

    def __init__(self, name="self_improving_agent", level=logging.INFO, log_dir="logs"):
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)

        # Create log directory structure
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)

        # Create subdirectories for different log types
        (log_path / "performance").mkdir(exist_ok=True)
        (log_path / "learning").mkdir(exist_ok=True)
        (log_path / "errors").mkdir(exist_ok=True)

        # Setup different log files
        self._setup_handlers(log_path)

        # Performance tracking
        self.performance_buffer = []

    def _setup_handlers(self, log_path: Path):
        """Setup various log handlers for different purposes."""

        # Remove existing handlers
        for handler in self._logger.handlers[:]:
            self._logger.removeHandler(handler)

        # Console handler with color coding
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = ColoredFormatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        self._logger.addHandler(console_handler)

        # Main log file (rotating)
        from logging.handlers import RotatingFileHandler
        main_handler = RotatingFileHandler(
            log_path / "agent.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        main_formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s"
        )
        main_handler.setFormatter(main_formatter)
        self._logger.addHandler(main_handler)

        # Error log file
        error_handler = logging.FileHandler(log_path / "errors" / f"errors_{datetime.now().strftime('%Y%m%d')}.log")
        error_handler.setLevel(logging.ERROR)
        error_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s\n%(exc_info)s"
        )
        error_handler.setFormatter(error_formatter)
        self._logger.addHandler(error_handler)

        # Performance log (JSON format for analysis)
        self.perf_handler = logging.FileHandler(
            log_path / "performance" / f"performance_{datetime.now().strftime('%Y%m%d')}.jsonl"
        )
        self.perf_logger = logging.getLogger(f"{self._logger.name}.performance")
        self.perf_logger.addHandler(self.perf_handler)
        self.perf_logger.setLevel(logging.INFO)

    def info(self, msg, *args, **kwargs):
        """Logs a message at the INFO level."""
        self._logger.info(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """Logs a message at the ERROR level."""
        self._logger.error(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        """Logs a message at the DEBUG level."""
        self._logger.debug(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """Logs a message at the WARNING level."""
        self._logger.warning(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        """Logs an exception at the ERROR level."""
        self._logger.exception(msg, *args, **kwargs)

    def isEnabledFor(self, level):
        """Check if a message of the given level would actually be logged."""
        return self._logger.isEnabledFor(level)

    def log(self, level, msg, *args, **kwargs):
        """Log a message with the specified level."""
        self._logger.log(level, msg, *args, **kwargs)

    def performance(self, metric: str, value: Any, context: Dict[str, Any] = None):
        """Log performance metrics in structured format."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "metric": metric,
            "value": value,
            "context": context or {}
        }
        self.perf_logger.info(json.dumps(log_entry))
        self.performance_buffer.append(log_entry)

    def learning(self, event: str, details: Dict[str, Any]):
        """Log learning events for analysis."""
        learning_logger = logging.getLogger(f"{self._logger.name}.learning")
        handler = logging.FileHandler(
            Path("logs") / "learning" / f"learning_{datetime.now().strftime('%Y%m%d')}.jsonl"
        )
        learning_logger.addHandler(handler)

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "details": details
        }
        learning_logger.info(json.dumps(log_entry))

    def close(self):
        """Close all handlers attached to the logger."""
        self.info("Closing logger and saving final metrics")

        # Save performance summary
        if self.performance_buffer:
            summary_path = Path("logs") / "performance" / "summary.json"
            with open(summary_path, 'w') as f:
                json.dump(self.performance_buffer, f, indent=2)

        for handler in self._logger.handlers[:]:
            handler.close()
            self._logger.removeHandler(handler)


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color coding for console output."""

    COLORS = {
        'DEBUG': '\033[36m',  # Cyan
        'INFO': '\033[32m',  # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',  # Red
        'CRITICAL': '\033[41m',  # Red background
    }
    RESET = '\033[0m'

    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        return super().format(record)