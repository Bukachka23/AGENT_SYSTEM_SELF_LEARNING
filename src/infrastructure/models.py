import json
from typing import Any, Optional

import google.generativeai as genai

from src.core.interfaces import AIModelInterface, StateStorageInterface


class GeminiAIModel(AIModelInterface):
    """Concrete implementation for Gemini AI"""

    def __init__(self, api_key: str, model_name: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate_content(self, prompt: str) -> Any:
        """Generate content using Gemini AI."""
        return self.model.generate_content(prompt)


class FileStateStorage(StateStorageInterface):
    """Concrete implementation for file-based storage"""

    def save(self, data: dict[str, Any], filepath: str) -> None:
        """Save state to file."""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str) -> Optional[dict[str, Any]]:
        """Load state from file if it exists."""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception:
            return None
