from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import List

from .auth import (
    create_access_token,
    decode_access_token,
    get_password_hash,
    verify_password,
)
from .db import get_db, is_database_available
from .model_loader import get_sentiment_model
from .schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    PredictionRecord,
    Token,
    UserCreate,
)
from .sentiment_utils import format_sentiment_label
from . import models

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

router = APIRouter()


def get_current_user_id(token: str = Depends(oauth2_scheme)) -> int:
    payload = decode_access_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication",
        )

    try:
        return int(payload.get("sub"))
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication",
        )


@router.get("/", response_model=HealthResponse)
def root():
    return {"status": "Sentiment Analysis API is running"}


@router.get("/health", response_model=HealthResponse)
def health(response: Response):
    database_ok = is_database_available()
    model_ok = get_sentiment_model().is_loaded()

    if database_ok and model_ok:
        return {"status": "ok", "database": "ok", "model": "ok"}

    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "degraded",
        "database": "ok" if database_ok else "unavailable",
        "model": "ok" if model_ok else "unavailable",
    }


@router.post("/predict", response_model=PredictResponse)
def predict(
    request: PredictRequest,
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    model = get_sentiment_model()
    try:
        details = model.predict_details(request.text)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    record = models.Prediction(
        user_id=user_id,
        text=request.text,
        sentiment=str(details["raw_label"]),
        confidence=details["confidence"],
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return {
        "sentiment": details["sentiment"],
        "confidence": details["confidence"],
        "top_terms": details["top_terms"],
    }


@router.get("/history", response_model=List[PredictionRecord])
def history(
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    records = (
        db.query(models.Prediction)
        .filter(models.Prediction.user_id == user_id)
        .order_by(models.Prediction.created_at.desc())
        .all()
    )
    for record in records:
        record.sentiment = format_sentiment_label(record.sentiment)
    return records


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    model = get_sentiment_model()
    metadata = model.get_metadata()
    if not metadata:
        return {"status": "unavailable"}

    evaluation = metadata.get("evaluation", {})
    dataset = metadata.get("dataset", {})
    vectorizer = metadata.get("vectorizer", {})

    return {
        "status": "available",
        "backend": metadata.get("backend"),
        "base_model": metadata.get("base_model"),
        "trained_at": metadata.get("trained_at"),
        "data_source": dataset.get("source"),
        "total_samples": dataset.get("total_samples"),
        "class_distribution": dataset.get("class_distribution", {}),
        "evaluation_accuracy": evaluation.get("accuracy"),
        "evaluation_macro_f1": evaluation.get("macro_f1"),
        "vectorizer_ngram_range": vectorizer.get("ngram_range", []),
        "training_max_length": metadata.get("training", {}).get("max_length"),
        "top_features_supported": hasattr(model.model, "coef_"),
    }


@router.post("/register", response_model=Token)
def register(user: UserCreate, db: Session = Depends(get_db)):
    existing = (
        db.query(models.User)
        .filter(models.User.username == user.username)
        .first()
    )
    if existing:
        raise HTTPException(
            status_code=400,
            detail="Username already registered",
        )
    hashed = get_password_hash(user.password)
    db_user = models.User(username=user.username, hashed_password=hashed)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    token = create_access_token({"sub": str(db_user.id)})
    return {"access_token": token, "token_type": "bearer"}


@router.post("/token", response_model=Token)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    user = (
        db.query(models.User)
        .filter(models.User.username == form_data.username)
        .first()
    )
    if not user or not verify_password(
        form_data.password,
        user.hashed_password,
    ):
        raise HTTPException(
            status_code=400,
            detail="Incorrect username or password",
        )
    token = create_access_token({"sub": str(user.id)})
    return {"access_token": token, "token_type": "bearer"}
