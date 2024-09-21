# app/core/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Hyperbolic LLM API"
    PROJECT_VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    DATABASE_URL: str = "sqlite:///./hyperbolic_llm.db"
    LLM_API_URL: str = "http://localhost:5000/generate" 
    class Config:
        env_file = ".env"

settings = Settings()
