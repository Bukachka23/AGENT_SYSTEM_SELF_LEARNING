

class StrategyAnalyzer:
    """Pure business logic for strategy analysis"""

    def _get_top_approaches(self, successful_strategies: list[dict]) -> list[str]:
        """Get top performing approaches from recent strategies."""
        approach_scores = {}
        for strategy in successful_strategies:
            approach = strategy['approach']
            score = strategy['score']
            if approach not in approach_scores:
                approach_scores[approach] = []
            approach_scores[approach].append(score)

        avg_approach_scores = {approach: sum(scores) / len(scores) for approach, scores in approach_scores.items()}
        sorted_approaches = sorted(avg_approach_scores.items(), key=lambda x: x[1], reverse=True)
        return [approach for approach, score in sorted_approaches[:3]]


class ApproachAnalyzer:
    """Pure business logic for determining solution approaches"""

    def _determine_approach(self, solution: str) -> str:
        """Determine the approach used in the solution."""
        approaches = {
            'iterative': ['for', 'while', 'loop'],
            'recursive': ['recursive', 'recursion', 'return self'],
            'functional': ['lambda', 'map', 'filter', 'reduce'],
            'object_oriented': ['class', 'self.', '__init__'],
            'algorithmic': ['algorithm', 'complexity', 'optimize']
        }

        solution_lower = solution.lower()
        for approach, keywords in approaches.items():
            if any(keyword in solution_lower for keyword in keywords):
                return approach
        return 'default'
