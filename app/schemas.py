from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field
from typing import Dict, List, Optional


class HealthResponse(BaseModel):
    status: str = "ok"
    database: Optional[str] = None
    model: Optional[str] = None


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)


class PredictResponse(BaseModel):
    sentiment: str
    confidence: float
    top_terms: List["TopTerm"] = Field(default_factory=list)


class UserCreate(BaseModel):
    username: str = Field(..., min_length=3)
    password: str = Field(..., min_length=6)


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class PredictionRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: Optional[int]
    text: str
    sentiment: str
    confidence: float
    created_at: datetime


class TopTerm(BaseModel):
    term: str
    contribution: float


class ModelInfoResponse(BaseModel):
    status: str
    backend: Optional[str] = None
    base_model: Optional[str] = None
    trained_at: Optional[str] = None
    data_source: Optional[str] = None
    total_samples: Optional[int] = None
    class_distribution: Dict[str, int] = Field(default_factory=dict)
    evaluation_accuracy: Optional[float] = None
    evaluation_macro_f1: Optional[float] = None
    vectorizer_ngram_range: List[int] = Field(default_factory=list)
    training_max_length: Optional[int] = None
    top_features_supported: bool = False
