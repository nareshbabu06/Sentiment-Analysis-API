import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
# Prefer the project-local .env during local runs so stale terminal variables
# from previous sessions do not override the intended app configuration.
load_dotenv(BASE_DIR / ".env", override=True)


def _as_bool(value: str, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_csv(value: str) -> list[str]:
    if not value:
        return []
    parts = [part.strip() for part in value.split(",")]
    return [part for part in parts if part]


class Settings:
    def __init__(self):
        self.app_env = os.getenv("APP_ENV", "development").strip().lower()
        self.database_url = os.getenv("DATABASE_URL", "sqlite:///./sentiment.db")
        self.jwt_secret = os.getenv("JWT_SECRET", "changemeplease")
        self.access_token_expire_minutes = int(
            os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440")
        )
        self.log_level = os.getenv("LOG_LEVEL", "INFO").strip().upper()
        self.host = os.getenv("HOST", "0.0.0.0").strip()
        self.port = int(os.getenv("PORT", "8080"))
        self.web_concurrency = max(1, int(os.getenv("WEB_CONCURRENCY", "1")))

        default_cors = "*" if self.app_env != "production" else ""
        self.cors_origins = _as_csv(os.getenv("CORS_ORIGINS", default_cors))

        require_model_default = self.app_env == "production"
        self.require_model_on_startup = _as_bool(
            os.getenv("REQUIRE_MODEL_ON_STARTUP"),
            default=require_model_default,
        )

        self.validate()

    def validate(self):
        if self.app_env == "production":
            if self.jwt_secret == "changemeplease":
                raise ValueError(
                    "JWT_SECRET must be set to a non-default value in production."
                )
            if self.database_url == "sqlite:///./sentiment.db":
                raise ValueError(
                    "DATABASE_URL must be set explicitly in production."
                )

    @property
    def allow_all_cors(self) -> bool:
        return self.cors_origins == ["*"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
