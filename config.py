"""MAFA Agents Configuration - Environment validation and settings.

Validates all required environment variables at startup (fail-fast pattern).
"""

import os
import sys
import logging
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings with validation."""
    
    # Required - Google AI
    GOOGLE_API_KEY: str = Field(..., description="Google Generative AI API key")
    
    # Required - Supabase
    SUPABASE_URL: str = Field(..., description="Supabase project URL")
    SUPABASE_API_KEY: str = Field(..., description="Supabase API key")
    
    # Required - Google Custom Search
    CUSTOM_SEARCH_API_KEY: str = Field(..., description="Google Custom Search API key")
    CUSTOM_SEARCH_CX: str = Field(..., description="Google Custom Search Engine ID")
    
    # Optional - External services
    BROKER_API_URL: str = Field(default="http://localhost:8080", description="Broker API URL")
    REDIS_URL: str = Field(default="redis://localhost:6379", description="Redis connection URL")
    
    # Optional - Fallback behavior
    USE_FALLBACK_DATA: bool = Field(default=True, description="Use fallback data when broker unavailable")
    
    # Optional - Security
    ALLOWED_ORIGINS: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        description="Comma-separated allowed CORS origins"
    )
    
    # Optional - Logging
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    
    @field_validator("GOOGLE_API_KEY")
    @classmethod
    def validate_google_key(cls, v: str) -> str:
        if not v or v in ("your_key_here", "your_gemini_api_key_here"):
            logger.error("GOOGLE_API_KEY not configured properly")
            sys.exit(1)
        return v
    
    @field_validator("SUPABASE_URL")
    @classmethod
    def validate_supabase_url(cls, v: str) -> str:
        if not v or "your-project" in v:
            logger.error("SUPABASE_URL not configured properly")
            sys.exit(1)
        return v
    
    @property
    def allowed_origins_list(self) -> list[str]:
        """Parse allowed origins into list."""
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


def get_settings() -> Settings:
    """Get validated settings instance."""
    try:
        return Settings()
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        logger.error("Please check your .env file and ensure all required variables are set.")
        sys.exit(1)


# Mock prices for development/testing when broker API is unavailable
MOCK_PRICES = {
    "AAPL": 195.50,
    "TSLA": 238.75,
    "GOOGL": 142.30,
    "MSFT": 415.20,
    "AMZN": 178.90,
    "NVDA": 722.48,
    "META": 485.60,
    "JPM": 198.45,
    "IBM": 167.80,
    "ORCL": 125.30,
    "ADBE": 545.20,
}

MOCK_HOLDINGS = {
    "AAPL": 10,
    "TSLA": 5,
    "GOOGL": 8,
    "MSFT": 15,
}

MOCK_BALANCE = 50000.0


# Load settings on import (fail-fast)
try:
    settings = get_settings()
except SystemExit:
    # Allow import to succeed for testing, but settings will be None
    settings = None  # type: ignore
