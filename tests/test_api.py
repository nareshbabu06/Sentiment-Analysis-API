import os
import sys
from uuid import uuid4

# Ensure project root is on PYTHONPATH so the app package is importable
# during tests
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), ".."),
    ),
)

from app.auth import decode_access_token  # noqa: E402
from app.db import Base, engine  # noqa: E402
from app.main import app  # noqa: E402
from app.preprocessing import preprocess_text  # noqa: E402
from fastapi.testclient import TestClient
import pytest

# Ensure DB tables exist for tests
Base.metadata.create_all(bind=engine)


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Sentiment" in r.json().get("status")


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["database"] == "ok"
    assert r.json()["model"] == "ok"


def test_preprocess_preserves_negation():
    assert preprocess_text("This is not good") == "not good"


def test_register_login_predict_history(client, monkeypatch):
    username = f"testuser_{uuid4().hex[:8]}"
    password = "testpass"
    r = client.post(
        "/register",
        json={"username": username, "password": password},
    )
    assert r.status_code == 200

    register_token = r.json()["access_token"]
    payload = decode_access_token(register_token)
    assert payload is not None, f"Token decode failed: {register_token}"

    r2 = client.post(
        "/token",
        data={"username": username, "password": password},
    )
    assert r2.status_code == 200

    token = r2.json()["access_token"]
    payload = decode_access_token(token)
    assert payload is not None, f"Token decode failed: {token}"

    class Dummy:
        def predict_details(self, text):
            return {
                "raw_label": 1,
                "sentiment": "Positive",
                "confidence": 0.99,
                "probabilities": {"Positive": 0.99, "Negative": 0.01},
                "top_terms": [{"term": "love", "contribution": 0.88}],
            }

    monkeypatch.setattr("app.routes.get_sentiment_model", lambda: Dummy())

    headers = {"Authorization": f"Bearer {token}"}
    r = client.post(
        "/predict",
        json={"text": "I love this"},
        headers=headers,
    )
    assert r.status_code == 200
    data = r.json()
    assert data["sentiment"] == "Positive"
    assert data["confidence"] > 0.9
    assert data["top_terms"][0]["term"] == "love"

    history = client.get("/history", headers=headers)
    assert history.status_code == 200
    records = history.json()
    assert len(records) == 1
    assert records[0]["text"] == "I love this"
    assert records[0]["sentiment"] == "Positive"


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] in {"available", "unavailable"}
