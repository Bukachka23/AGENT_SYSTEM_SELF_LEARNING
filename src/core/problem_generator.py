class ProblemGenerator:
    """Generate problems on-demand instead of using storing them all"""
    def __init__(self):
        self.problem_templates = [
            ("factorial", "Write a function to calculate the factorial of a number"),
            ("calculator", "Create a simple text-based calculator that handles basic operations"),
            ("pathfinding", "Design a system to find the shortest path between two points in a graph"),
            ("recommendation", "Implement a basic recommendation system for movies based on user preferences"),
            ("ml_prediction", "Create a machine learning model to predict house prices based on features")
        ]

    def get_problem(self, index):
        """Get a specific problem by index."""
        if 0 <= index < len(self.problem_templates):
            return self.problem_templates[index][1]
        return None

    def iterate_problems(self):
        """Generator to iterate trough problems without loading all at once."""
        for _, problem in self.problem_templates:
            yield problem
