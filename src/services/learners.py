from typing import Any


class CapabilityLearner:
    """Pure business logic for capability improvement"""

    def calculate_capability_improvements(self, capabilities: dict[str, float], avg_score: float) -> dict[str, float]:
        """Calculate new capability values based on performance"""
        improvement_factor = 0.05 * avg_score
        new_capabilities = {}

        for capability, current_value in capabilities.items():
            new_capabilities[capability] = min(1.0, current_value + improvement_factor)

        return new_capabilities

    def calculate_metrics(self, results: list[Any]) -> dict[str, float]:
        """Calculate performance metrics from results"""
        if not results:
            return {'avg_score': 0.0, 'success_rate': 0.0}

        successful_results = [r for r in results if r.quality_score > 0.7]
        avg_score = sum(r.quality_score for r in results) / len(results)
        success_rate = len(successful_results) / len(results)

        return { 'avg_score': avg_score, 'success_rate': success_rate, 'successful_count': len(successful_results)}