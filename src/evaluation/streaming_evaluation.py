
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd


BASE_DIR = Path.cwd()

REPORT_DIR = BASE_DIR / "src" / "results" / "streaming" / "reports"
BATCH_DIR = BASE_DIR / "src" / "results" / "streaming" / "batches"

OUTPUT_DIR = BASE_DIR / "results" / "streaming" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 1. Load monthly topic reports
# --------------------------------------------------

def load_reports():

    files = sorted(REPORT_DIR.glob("*_topics.parquet"))

    if not files:
        raise FileNotFoundError(
            f"No monthly topic reports found in {REPORT_DIR}"
        )

    reports = []

    for file in files:

        df = pd.read_parquet(file)

        df["month"] = file.stem.replace("_topics", "")

        reports.append(df)

    data = pd.concat(reports, ignore_index=True)

    data["topic_id"] = data["topic_id"].astype(int)

    print(f"Loaded {len(data):,} articles.")
    print(f"Loaded {len(files)} monthly reports.")

    return data


# --------------------------------------------------
# 2. Topic activity across months
# --------------------------------------------------

def calculate_topic_activity(data):

    activity = (
        data[data["topic_id"] != -1]
        .groupby(["month", "topic_id"])
        .size()
        .reset_index(name="article_count")
    )

    return activity


# --------------------------------------------------
# 3. Topic stability
# --------------------------------------------------

def calculate_topic_stability(activity):

    stability_records = []

    months = sorted(activity["month"].unique())

    for i in range(1, len(months)):

        previous_month = months[i - 1]
        current_month = months[i]

        previous = activity[
            activity["month"] == previous_month
        ]

        current = activity[
            activity["month"] == current_month
        ]

        previous_topics = set(previous["topic_id"])
        current_topics = set(current["topic_id"])

        union = previous_topics | current_topics
        intersection = previous_topics & current_topics

        jaccard = (
            len(intersection) / len(union)
            if union else 1.0
        )

        stability_records.append({
            "previous_month": previous_month,
            "current_month": current_month,
            "shared_topics": len(intersection),
            "total_unique_topics": len(union),
            "topic_jaccard_stability": jaccard
        })

    return pd.DataFrame(stability_records)


# --------------------------------------------------
# 4. Emerging topic statistics
# --------------------------------------------------

def calculate_emerging_statistics(data):

    emerging = data[data["topic_type"] == "emerging"]

    monthly = (
        emerging.groupby("month")
        .agg(
            emerging_articles=("id", "count"),
            emerging_topics=("topic_id", "nunique")
        )
        .reset_index()
    )

    total_articles = len(data)

    summary = {
        "total_emerging_articles": int(len(emerging)),
        "total_emerging_topics": int(emerging["topic_id"].nunique()),
        "emerging_article_percentage": (
            round(len(emerging) / total_articles * 100, 2)
            if total_articles else 0
        )
    }

    return monthly, summary


# --------------------------------------------------
# 5. Topic growth detection
# --------------------------------------------------

def calculate_topic_growth(activity):

    pivot = activity.pivot_table(
        index="month",
        columns="topic_id",
        values="article_count",
        fill_value=0
    ).sort_index()

    growth_records = []

    for topic_id in pivot.columns:

        counts = pivot[topic_id]

        for i in range(1, len(counts)):

            previous = int(counts.iloc[i - 1])
            current = int(counts.iloc[i])

            growth_percentage = (
                ((current - previous) / previous) * 100
                if previous > 0 else None
            )

            growth_records.append({
                "month": counts.index[i],
                "topic_id": int(topic_id),
                "previous_articles": previous,
                "current_articles": current,
                "growth_percentage": growth_percentage
            })

    return pd.DataFrame(growth_records)


# --------------------------------------------------
# 6. Processing performance
# --------------------------------------------------

def calculate_processing_performance():

    files = sorted(BATCH_DIR.glob("*_summary.json"))

    records = []

    for file in files:

        with open(file, "r", encoding="utf-8") as f:
            summary = json.load(f)

        article_count = int(
            summary.get("article_count", 0)
        )

        processing_seconds = summary.get(
            "processing_seconds"
        )

        records.append({
            "month": file.stem.replace("_summary", ""),
            "article_count": article_count,
            "processing_seconds": processing_seconds,
            "articles_per_second": (
                article_count / processing_seconds
                if processing_seconds and processing_seconds > 0
                else None
            )
        })

    return pd.DataFrame(records)


# --------------------------------------------------
# 7. Main evaluation
# --------------------------------------------------

def run_evaluation():

    start_time = time.perf_counter()

    print("Loading reports...")

    data = load_reports()

    print("Calculating topic activity...")

    activity = calculate_topic_activity(data)

    print("Calculating topic stability...")

    stability = calculate_topic_stability(activity)

    print("Calculating emerging topic statistics...")

    emerging_monthly, emerging_summary = (
        calculate_emerging_statistics(data)
    )

    print("Calculating topic growth...")

    growth = calculate_topic_growth(activity)

    print("Calculating processing performance...")

    performance = calculate_processing_performance()

    # Save detailed results.

    activity.to_csv(
        OUTPUT_DIR / "topic_activity.csv",
        index=False
    )

    stability.to_csv(
        OUTPUT_DIR / "topic_stability.csv",
        index=False
    )

    emerging_monthly.to_csv(
        OUTPUT_DIR / "emerging_topics.csv",
        index=False
    )

    growth.to_csv(
        OUTPUT_DIR / "topic_growth.csv",
        index=False
    )

    performance.to_csv(
        OUTPUT_DIR / "processing_performance.csv",
        index=False
    )

    total_time = time.perf_counter() - start_time

    summary = {
        "total_articles": int(len(data)),
        "total_months": int(data["month"].nunique()),
        "total_topics": int(data.loc[
            data["topic_id"] != -1, "topic_id"
        ].nunique()),
        "unassigned_articles": int(
            (data["topic_id"] == -1).sum()
        ),
        "average_topic_stability": (
            round(
                float(stability["topic_jaccard_stability"].mean()),
                4
            )
            if not stability.empty else None
        ),
        **emerging_summary,
        "evaluation_runtime_seconds": round(total_time, 2)
    }

    with open(
        OUTPUT_DIR / "evaluation_summary.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(summary, f, indent=4)

    print("\nEvaluation completed.")
    print(json.dumps(summary, indent=4))

    print(f"\nResults saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_evaluation()
