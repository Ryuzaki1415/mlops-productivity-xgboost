from pydantic import BaseModel, Field, field_validator
from typing import Literal
from typing import Optional


# ── Input schema ───────────────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    # Raw numeric inputs
    Daily_Phone_Hours:          float = Field(..., ge=0, le=24,  description="Hours/day on phone")
    Social_Media_Hours:         float = Field(..., ge=0, le=24,  description="Hours/day on social media")
    Sleep_Hours:                float = Field(..., ge=0, le=24,  description="Hours of sleep per night")
    Stress_Level:               float = Field(..., ge=1, le=10,  description="Stress level 1–10")
    App_Usage_Count:            int   = Field(..., ge=1, le=100, description="Number of apps used daily")
    Caffeine_Intake_Cups:       float = Field(..., ge=0, le=20,  description="Cups of caffeine per day")
    Weekend_Screen_Time_Hours:  float = Field(..., ge=0, le=24,  description="Weekend screen hours/day")

    # Categorical inputs
    Gender:      Literal["Male", "Female", "Non-binary"]
    Occupation:  Literal["Engineer", "Student", "Manager", "Healthcare", "Creative", "Sales"]
    Device_Type: Literal["Android", "iOS", "Both"]

    @field_validator("Social_Media_Hours")
    @classmethod
    def social_cannot_exceed_phone(cls, v, info):
        phone = info.data.get("Daily_Phone_Hours")
        if phone is not None and v > phone:
            raise ValueError("Social_Media_Hours cannot exceed Daily_Phone_Hours")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "Daily_Phone_Hours": 7.5,
                "Social_Media_Hours": 3.0,
                "Sleep_Hours": 6.5,
                "Stress_Level": 7.0,
                "App_Usage_Count": 20,
                "Caffeine_Intake_Cups": 2.5,
                "Weekend_Screen_Time_Hours": 9.0,
                "Gender": "Male",
                "Occupation": "Engineer",
                "Device_Type": "iOS"
            }
        }


# ── SHAP contributor schema ────────────────────────────────────────────────────
class SHAPContributor(BaseModel):
    feature:    str
    value:      float   # actual feature value
    shap_value: float   # SHAP contribution to prediction


# ── Prediction response ────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    score:            float
    score_category:   Literal["Low", "Moderate", "High"]
    top_contributors: list[SHAPContributor]
    insight:          str  # LLM-generated explanation


# ── Health check ──────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status:      str
    model_loaded: bool
    ollama_ready: bool
