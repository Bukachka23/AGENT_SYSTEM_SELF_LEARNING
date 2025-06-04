import os

from dotenv import load_dotenv


class Config:
    """Configuration class for loading environment variables."""

    def __init__(self):
        load_dotenv()

        self.google_api_key: str | None = os.getenv("GOOGLE_API_KEY")
