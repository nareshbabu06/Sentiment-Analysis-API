# Sentiment Analysis API

A FastAPI-based sentiment analysis REST API with PostgreSQL storage, JWT authentication, Docker support, and unit tests.

## Features

- Sentiment classification using TF-IDF + Logistic Regression
- Better phrase understanding with unigram + bigram TF-IDF features
- Model metadata and evaluation report saved after training
- Prediction explanations via top contributing terms
- NLTK text preprocessing (lowercase, punctuation removal, stop word removal, tokenization, lemmatization)
- PostgreSQL prediction history storage
- User registration and login with JWT tokens
- Docker and Docker Compose for deployment
- GitHub Actions CI for tests

## Setup

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Train the model:

```bash
python training/train.py --data path/to/your.csv
```

Optional tuning:

```bash
python training/train.py --data path/to/your.csv --max-features 8000 --ngram-max 2
```

3. Copy the local environment file:

```bash
copy .env.example .env
```

4. Start the API:

```bash
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8080
```

5. Verify the API:

```bash
curl http://127.0.0.1:8080/health
```

6. Use Swagger UI in a regular browser:

http://127.0.0.1:8080/docs

## Postman

- Register: POST `/register`
- Login: POST `/token` as `application/x-www-form-urlencoded` with `username` and `password`
- Predict: POST `/predict` with `Authorization: Bearer <token>`
- History: GET `/history` with `Authorization: Bearer <token>`
- Model info: GET `/model-info`

## Deployment

The deployment image expects the trained artifacts in `models/sentiment_model.pkl` and `models/vectorizer.pkl`. Keep those files in the repo when you deploy.

1. Copy the Docker environment file:

```bash
copy .env.docker.example .env
```

2. Update `.env` with a real `JWT_SECRET` and allowed `CORS_ORIGINS`.
3. Build and start the stack:

```bash
docker compose up --build -d
```

4. Verify the deployment:

```bash
curl http://localhost:8080/health
```

The service is ready when `/health` returns `{"status":"ok","database":"ok","model":"ok"}`.

## Environment

Important variables:

- `APP_ENV=production`
- `DATABASE_URL`
- `JWT_SECRET`
- `CORS_ORIGINS` as a comma-separated list when you have multiple frontends
- `WEB_CONCURRENCY` to scale Uvicorn workers
