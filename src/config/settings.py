from pydantic import BaseModel


from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class PerformanceMetrics:
    """Lightweight dataclass for performance metrics"""
    timestamp: float
    cycle: int
    avg_score: float
    memory_used: float
    success_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'cycle': self.cycle,
            'avg_score': self.avg_score,
            'memory_used': self.memory_used,
            'success_rate': self.success_rate
        }


@dataclass
class SolutionResult:
    """Lightweight result container"""
    problem: str
    solution: str
    quality_score: float
    execution_time: float
    memory_used: float
    approach: str = "default"
    error: Optional[str] = None

    def __hash__(self):
        return hash(self.problem)

@dataclass
class AgentState:
    capabilities: dict[str, float]
    total_problems_solved: int
    current_cycle: int
    learned_patterns: list[dict[str, Any]]
    successful_strategies: list[dict[str, Any]]
    stats: dict[str, int]
    timestamp: float


class LLMConfig(BaseModel):
    model: str = 'gemini-2.5-flash-preview-05-20'
    problem_solving: float= 0.5
    code_generation: float = 0.5
    learning_efficiency: float = 0.5
    error_handling: float = 0.5
