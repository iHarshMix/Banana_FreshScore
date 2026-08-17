"""Pydantic schemas for FastAPI serving requests and responses."""

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """Schema for banana ripeness inference prediction response."""

    spoilage_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Continuous spoilage/ripeness score between 0.0 (unripe) and 1.0 (rotten)",
        json_schema_extra={"example": 0.3540},
    )
    category: str = Field(
        ...,
        description=(
            "Qualitative ripeness category (Unripe, Slightly Ripe, Ripe, Overripe / Rotten)"
        ),
        json_schema_extra={"example": "Slightly Ripe"},
    )
    shelf_life_days: float = Field(
        ...,
        ge=0.0,
        le=12.0,
        description="Estimated remaining shelf-life in days",
        json_schema_extra={"example": 7.75},
    )
    recommended_action: str = Field(
        ...,
        description="Operational warehouse or retail recommended action",
        json_schema_extra={"example": "Ready for retail distribution"},
    )
    model_version: str = Field(
        default="v1.0.0",
        description="Active model identifier",
        json_schema_extra={"example": "v1.0.0"},
    )
    latency_ms: float = Field(
        ...,
        description="Inference latency in milliseconds",
        json_schema_extra={"example": 35.2},
    )


class HealthResponse(BaseModel):
    """Schema for health probe response."""

    status: str = Field(
        default="healthy", json_schema_extra={"example": "healthy"}
    )
    model_loaded: bool = Field(
        default=True, json_schema_extra={"example": True}
    )
    model_path: str = Field(
        default="models/production_model.pt",
        json_schema_extra={"example": "models/production_model.pt"},
    )
    device: str = Field(default="cpu", json_schema_extra={"example": "cpu"})
