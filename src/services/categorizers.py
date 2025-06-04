class ProblemCategorizer:
    """Pure business logic for categorizing problems"""

    def _categorize_problem(self, problem: str) -> str:
        """Categorize problem type for learning."""
        categories = {
            'algorithm': ['sort', 'search', 'optimize', 'algorithm'],
            'data_structure': ['array', 'list', 'tree', 'graph', 'stack'],
            'system_design': ['design', 'system', 'architecture', 'scale'],
            'machine_learning': ['predict', 'model', 'ml', 'machine learning'],
            'general': []
        }

        problem_lower = problem.lower()
        for category, keywords in categories.items():
            if any(keyword in problem_lower for keyword in keywords):
                return category
        return 'general'

