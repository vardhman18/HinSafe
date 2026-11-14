
from typing import Literal

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Data accepted by the comment prediction endpoint."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Hinglish comment to analyze.",
    )
    explain: bool = Field(
        True,
        description="Set to false to skip SHAP word highlighting (faster).",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": (
                    "“tu itna ghatiya insaan hai, teri shakal dekh ke hi ulti aati hai, jaake mar ja kahin”"
                ),
                "explain": True,
            }
        }
    }


class PredictResponse(BaseModel):
    """Result returned by the comment prediction endpoint."""

    comment: str = Field(description="The original comment.")
    label: Literal[0, 1] = Field(
        description="Classification label: 0 is not offensive and 1 is offensive.",
    )
    severity_score: int = Field(
        ge=0,
        le=100,
        description="Offensive probability scaled to a score from 0 to 100.",
    )
    severity_band: Literal["SAFE", "OFFENSIVE", "HATE SPEECH"] = Field(
        description="Human-readable severity band for the score.",
    )
    categories: list[str] = Field(
        default_factory=list,
        description="Hate-speech categories detected by the keyword rules.",
    )
    flagged_words: dict[str, float] = Field(
        default_factory=dict,
        description="Words and phrases with positive SHAP contributions.",
    )


class HealthResponse(BaseModel):
    """Status returned by the health-check endpoint."""

    status: Literal["ok"] = Field(
        default="ok",
        description="Indicates that the API process is responding.",
    )
    model_loaded: bool = Field(
        description="Whether the prediction model has finished loading.",
    )
