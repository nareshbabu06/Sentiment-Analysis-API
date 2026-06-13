from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .routes import router as api_router
from .model_loader import get_sentiment_model
from .db import Base, engine

settings = get_settings()

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Creating DB tables if not exist")
    Base.metadata.create_all(bind=engine)
    try:
        logger.info("Loading model and vectorizer")
        get_sentiment_model().load()
    except Exception as exc:
        if settings.require_model_on_startup:
            raise RuntimeError("Failed to load the sentiment model at startup") from exc
        logger.warning("Model load failed at startup: %s", exc)
    yield


app = FastAPI(title="Sentiment Analysis API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins or [],
    allow_credentials=not settings.allow_all_cors,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(api_router)
