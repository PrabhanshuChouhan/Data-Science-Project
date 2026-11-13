
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "frontend" / "public" / "data"


# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------

def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV if it exists; otherwise return an empty DataFrame."""
    if not path.exists():
        print(f"[SKIP] Missing: {path.relative_to(BASE_DIR)}")
        return pd.DataFrame()

    return pd.read_csv(path)


def read_parquet(path: Path) -> pd.DataFrame:
    """Read a Parquet file if it exists; otherwise return an empty DataFrame."""
    if not path.exists():
        print(f"[SKIP] Missing: {path.relative_to(BASE_DIR)}")
        return pd.DataFrame()

    return pd.read_parquet(path)


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object if it exists."""
    if not path.exists():
        print(f"[SKIP] Missing: {path.relative_to(BASE_DIR)}")
        return {}

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


# ---------------------------------------------------------
# JSON CONVERSION
# ---------------------------------------------------------

def clean_value(value: Any) -> Any:
    """Convert NumPy/Pandas values into JSON-safe Python values."""

    if value is None:
        return None

    if isinstance(value, dict):
        return {
            str(key): clean_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [clean_value(item) for item in value]

    if isinstance(value, np.ndarray):
        return clean_value(value.tolist())

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None

    if isinstance(value, (np.bool_, bool)):
        return bool(value)

    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()

    if isinstance(value, (pd.Timedelta,)):
        return str(value)

    if isinstance(value, float):
        return value if np.isfinite(value) else None

    if isinstance(value, (str, int)):
        return value

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return str(value)


def dataframe_records(
    df: pd.DataFrame,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Convert a DataFrame into JSON-safe records."""

    if df.empty:
        return []

    data = df.head(limit) if limit is not None else df
    records = data.to_dict(orient="records")

    return clean_value(records)


def write_json(filename: str, data: Any) -> None:
    """Write a JSON file into the frontend public data directory."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / filename

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            clean_value(data),
            file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )

    print(f"[OK] Exported {output_path.relative_to(BASE_DIR)}")


# ---------------------------------------------------------
# OVERVIEW
# ---------------------------------------------------------

def export_overview() -> None:
    articles = read_parquet(PROCESSED_DIR / "news.parquet")

    topics = read_parquet(
        RESULTS_DIR / "topics" / "article_topics.parquet"
    )

    graph_metrics = read_parquet(
        RESULTS_DIR / "graph" / "article_graph_metrics.parquet"
    )

    category_count = (
        int(articles["category"].nunique())
        if not articles.empty and "category" in articles.columns
        else 0
    )

    date_min = None
    date_max = None

    if not articles.empty and "date" in articles.columns:
        dates = pd.to_datetime(
            articles["date"],
            errors="coerce",
        ).dropna()

        if not dates.empty:
            date_min = dates.min().date().isoformat()
            date_max = dates.max().date().isoformat()

    topic_count = 0

    if not topics.empty:
        for column in ("kmeans_cluster", "gmm_cluster"):
            if column in topics.columns:
                topic_count = max(
                    topic_count,
                    int(topics[column].nunique()),
                )

    total_windows = 0

    if not articles.empty:
        if "month" in articles.columns:
            total_windows = int(articles["month"].nunique())
        elif "date" in articles.columns:
            total_windows = int(
                pd.to_datetime(
                    articles["date"],
                    errors="coerce",
                ).dt.to_period("M").nunique()
            )

    overview = {
        "total_articles": int(len(articles)),
        "total_categories": category_count,
        "total_windows": total_windows,
        "total_topics": topic_count,
        "graph_nodes": int(len(graph_metrics)),
        "date_range": {
            "start": date_min,
            "end": date_max,
        },
    }

    write_json("overview.json", overview)


# ---------------------------------------------------------
# TREND ANALYSIS
# ---------------------------------------------------------

def export_trends() -> None:
    trend_dir = RESULTS_DIR / "trends"

    files = {
        "category_shares.json": "category_shares.csv",
        "category_drift.json": "category_drift.csv",
        "semantic_drift.json": "semantic_drift.csv",
        "topic_prevalence.json": "topic_prevalence.csv",
    }

    for output_name, input_name in files.items():
        df = read_csv(trend_dir / input_name)

        write_json(
            output_name,
            dataframe_records(df),
        )


# ---------------------------------------------------------
# TOPIC ANALYSIS
# ---------------------------------------------------------

def export_topics() -> None:
    topic_dir = RESULTS_DIR / "topics"

    keywords = read_csv(
        topic_dir / "topic_keywords.csv"
    )

    examples = read_csv(
        topic_dir / "topic_examples.csv"
    )

    metrics = read_csv(
        topic_dir / "clustering_metrics.csv"
    )

    write_json(
        "topic_keywords.json",
        dataframe_records(keywords),
    )

    write_json(
        "topic_examples.json",
        dataframe_records(examples, limit=100),
    )

    write_json(
        "clustering_metrics.json",
        dataframe_records(metrics),
    )


# ---------------------------------------------------------
# GRAPH ANALYSIS
# ---------------------------------------------------------

def export_graph() -> None:
    graph_dir = RESULTS_DIR / "graph"

    communities = read_csv(
        graph_dir / "community_summary.csv"
    )

    article_metrics = read_parquet(
        graph_dir / "article_graph_metrics.parquet"
    )

    # Rank articles by the strongest available centrality measure.
    if not article_metrics.empty:

        rank_column = next(
            (
                column
                for column in (
                    "pagerank",
                    "degree_centrality",
                    "weighted_degree",
                )
                if column in article_metrics.columns
            ),
            None,
        )

        if rank_column is not None:
            article_metrics = article_metrics.sort_values(
                rank_column,
                ascending=False,
            )

    write_json(
        "communities.json",
        dataframe_records(communities, limit=200),
    )

    write_json(
        "central_articles.json",
        dataframe_records(article_metrics, limit=100),
    )


# ---------------------------------------------------------
# EVENT / CAUSAL ANALYSIS
# ---------------------------------------------------------

def export_causal() -> None:
    causal_dir = RESULTS_DIR / "causal"

    # These are the actual outputs produced by src/causal.py.
    event_impact = read_csv(
        causal_dir / "event_impact.csv"
    )

    event_summary = read_csv(
        causal_dir / "event_summary.csv"
    )

    write_json(
        "event_impact.json",
        dataframe_records(event_impact),
    )

    write_json(
        "event_summary.json",
        dataframe_records(event_summary),
    )


# ---------------------------------------------------------
# PRIVACY ANALYSIS
# ---------------------------------------------------------

def export_privacy() -> None:
    privacy_dir = RESULTS_DIR / "privacy"

    private_stats = read_csv(
        privacy_dir / "private_category_statistics.csv"
    )

    metadata = read_json(
        privacy_dir / "privacy_metadata.json"
    )

    write_json(
        "private_statistics.json",
        dataframe_records(private_stats),
    )

    write_json(
        "privacy_metadata.json",
        metadata,
    )

    # Do not export results/privacy/internal/.
    # Internal utility evaluation is not part of the public dashboard.


# ---------------------------------------------------------
# FAIRNESS / REPRESENTATION ANALYSIS
# ---------------------------------------------------------

def export_fairness() -> None:
    fairness_dir = RESULTS_DIR / "fairness"

    summary = read_json(
        fairness_dir / "fairness_summary.json"
    )

    author_concentration = read_csv(
        fairness_dir / "author_concentration.csv"
    )

    category_by_cohort = read_csv(
        fairness_dir / "category_by_author_cohort.csv"
    )

    category_disparity = read_csv(
        fairness_dir / "category_disparity.csv"
    )

    cohort_divergence = read_csv(
        fairness_dir / "cohort_distribution_divergence.csv"
    )

    write_json(
        "fairness_summary.json",
        summary,
    )

    write_json(
        "author_concentration.json",
        dataframe_records(author_concentration),
    )

    write_json(
        "category_by_author_cohort.json",
        dataframe_records(category_by_cohort),
    )

    write_json(
        "category_disparity.json",
        dataframe_records(category_disparity),
    )

    write_json(
        "cohort_distribution_divergence.json",
        dataframe_records(cohort_divergence),
    )


# ---------------------------------------------------------
# MAIN EXPORT
# ---------------------------------------------------------

def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n" + "=" * 55)
    print("EXPORTING PIPELINE RESULTS FOR THE DASHBOARD")
    print("=" * 55 + "\n")

    export_overview()
    export_topics()
    export_trends()
    export_graph()
    export_causal()
    export_privacy()
    export_fairness()

    # Generate manifest after all other exports.
    exported_files = sorted(
        path.name
        for path in OUTPUT_DIR.glob("*.json")
        if path.name != "manifest.json"
    )

    manifest = {
        "generated_from": "Data-Science-Project pipeline outputs",
        "files": exported_files,
        "file_count": len(exported_files),
    }

    write_json(
        "manifest.json",
        manifest,
    )

    print("\n" + "=" * 55)
    print("DASHBOARD DATA EXPORT COMPLETE")
    print("=" * 55)

    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"JSON files exported: {len(exported_files) + 1}")


if __name__ == "__main__":
    main()
