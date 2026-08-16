from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central 12-factor application settings."""

    # Model & Storage
    model_path: str = "models/production_model.pt"
    mlflow_tracking_uri: str = "http://localhost:5000"
    s3_bucket: str = "banana-mlops-bucket"

    # Networking & Serving
    fastapi_port: int = 8000
    fastapi_host: str = "0.0.0.0"
    streamlit_port: int = 8501

    # Business Logic
    max_shelf_life_days: float = 12.0

    # Logging
    log_level: str = "INFO"

    # Data Ingestion (Kaggle)
    kaggle_username: str = ""
    kaggle_key: str = ""

    # AWS Infrastructure (Phase 2 / Phase 3)
    aws_region: str = "us-east-1"
    aws_account_id: str = ""
    ec2_public_ip: str = ""
    ec2_instance_id: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton instance of application settings."""
    return Settings()
