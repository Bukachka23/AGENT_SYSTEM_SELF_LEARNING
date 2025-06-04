
class SolutionEvaluator:
    """Pure business logic for evaluating solutions"""

    def extract_rating_from_response(self, response_text: str) -> float:
        """Extract rating from AI response text"""
        rating_text = response_text.lower()
        if 'rating:' in rating_text:
            rating_str = rating_text.split('rating:')[1].strip().split()[0]
            try:
                return float(rating_str)
            except ValueError:
                return 0.5
        return 0.5