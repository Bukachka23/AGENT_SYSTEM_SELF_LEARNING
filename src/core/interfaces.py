from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class AIModelInterface(ABC):
    @abstractmethod
    def generate_content(self, prompt: str) -> Any:
        pass


class StateStorageInterface(ABC):
    @abstractmethod
    def save(self, data: Dict[str, Any], filepath: str) -> None:
        pass

    @abstractmethod
    def load(self, filepath: str) -> Optional[Dict[str, Any]]:
        pass


class SystemMonitorInterface(ABC):
    @abstractmethod
    def get_process_memory_mb(self) -> float:
        pass

    @abstractmethod
    def get_cpu_percent(self) -> float:
        pass

    @abstractmethod
    def get_memory_info(self) -> Any:
        pass


class LoggerInterface(ABC):
    @abstractmethod
    def info(self, message: str, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def debug(self, message: str, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def error(self, message: str, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def warning(self, message: str, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def exception(self, message: str, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def performance(self, metric: str, value: Any, context: Dict[str, Any] = None) -> None:
        pass

    @abstractmethod
    def learning(self, event: str, details: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def close(self):
        pass
