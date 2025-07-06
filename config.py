"""
Configuration settings for the Autonomous Developer Agent
Uses Pydantic to load settings from environment variables
"""

from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """
    Main configuration settings for the agent
    """
    model: str = Field(
        default="codellama:latest",
        description="Ollama model to use for code generation",
        env="OLLAMA_MODEL"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# Global settings instance
settings = Settings()
