
import json
import os
from pathlib import Path

import pandas as pd

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

ANALYSIS_DIR = BASE_DIR / "results" / "streaming" / "analysis"
EVALUATION_DIR = BASE_DIR / "results" / "streaming" / "evaluation"

CATEGORY_DIR = (
    BASE_DIR / "src" / "results" / "streaming" / "analysis"
)

# Override with comma-separated frontend origins when deploying.
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]

app = FastAPI(
    title="News Stream Intelligence API",
    description="API for streaming topic discovery and trend analysis",
    version="1.0.0"
)

# Allow the Next.js frontend to communicate with this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return pd.read_csv(path).replace({float("nan"): None}).to_dict(
        orient="records"
    )


@app.get("/")
def home():
    return {
        "message": "News Stream Intelligence API is running",
        "version": "1.0.0"
    }


@app.get("/api/health")
def health():
    return {"status": "healthy"}


@app.get("/api/overview")
def overview():
    return load_json(EVALUATION_DIR / "evaluation_summary.json")


@app.get("/api/topics/activity")
def topic_activity():
    return load_csv(EVALUATION_DIR / "topic_activity.csv")


@app.get("/api/topics/growth")
def topic_growth():
    return load_csv(EVALUATION_DIR / "topic_growth.csv")


@app.get("/api/topics/stability")
def topic_stability():
    return load_csv(EVALUATION_DIR / "topic_stability.csv")


@app.get("/api/topics/emerging")
def emerging_topics():
    return load_csv(EVALUATION_DIR / "emerging_topics.csv")


@app.get("/api/trends/categories")
def category_trends():
    return load_csv(CATEGORY_DIR / "category_trends.csv")


@app.get("/api/analysis/semantic-drift")
def semantic_drift():
    return load_csv(ANALYSIS_DIR / "semantic_drift.csv")


@app.get("/api/analysis/network")
def article_network():
    return load_json(ANALYSIS_DIR / "article_network.json")


@app.get("/api/analysis/influential-articles")
def influential_articles():
    return load_csv(
        ANALYSIS_DIR / "top_influential_articles.csv"
    )


@app.get("/api/evaluation/performance")
def processing_performance():
    return load_csv(
        EVALUATION_DIR / "processing_performance.csv"
    )
