
"""
Temporal trend detection and semantic drift analysis.

Analysis:
    - Monthly category proportions
    - Two-proportion statistical tests
    - Benjamini-Hochberg correction
    - Semantic drift using embedding distributions
    - Topic prevalence over time

Optional experiment:
    - Evaluate detection using artificially injected trends

Inputs:
    data/processed/news.parquet
    data/processed/emb.npy
    results/topics/article_topics.parquet
    results/ground_truth.json (optional)

Outputs:
    results/trends/category_shares.csv
    results/trends/category_drift.csv
    results/trends/semantic_drift.csv
    results/trends/topic_prevalence.csv
    results/trends/category_drift_injected.csv
    results/trends/detection_metrics.json
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.spatial.distance import jensenshannon
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "data" / "processed"
TOPICS_DIR = BASE_DIR / "results" / "topics"
OUTPUT_DIR = BASE_DIR / "results" / "trends"

DATA_PATH = PROCESSED_DIR / "news.parquet"
EMBEDDINGS_PATH = PROCESSED_DIR / "emb.npy"
TOPICS_PATH = TOPICS_DIR / "article_topics.parquet"

# Matches the output path in the supplied stream.py.
GROUND_TRUTH_PATH = BASE_DIR / "results" / "ground_truth.json"


# ---------------------------------------------------------
# 1. LOAD AND VALIDATE DATA
# ---------------------------------------------------------

def load_data():

    required_files = [
        DATA_PATH,
        EMBEDDINGS_PATH,
        TOPICS_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    df = pd.read_parquet(DATA_PATH)

    embeddings = np.load(
        EMBEDDINGS_PATH,
        mmap_mode="r",
    )

    topics = pd.read_parquet(TOPICS_PATH)

    required_article_columns = {
        "id", "date", "category"
    }

    required_topic_columns = {
        "id", "kmeans_cluster"
    }

    if not required_article_columns.issubset(df.columns):
        raise ValueError(
            f"Article dataset needs columns: "
            f"{required_article_columns}"
        )

    if not required_topic_columns.issubset(topics.columns):
        raise ValueError(
            f"Topic assignments need columns: "
            f"{required_topic_columns}"
        )

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D matrix.")

    if df["id"].duplicated().any():
        raise ValueError("Duplicate article IDs found.")

    if topics["id"].duplicated().any():
        raise ValueError("Duplicate topic assignment IDs found.")

    df["id"] = df["id"].astype(int)
    topics["id"] = topics["id"].astype(int)

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    if df["date"].isna().any():
        raise ValueError("Invalid article dates found.")

    if df["category"].isna().any():
        raise ValueError("Missing article categories found.")

    if len(df) == 0:
        raise ValueError("Article dataset is empty.")

    if df["id"].min() < 0 or df["id"].max() >= embeddings.shape[0]:
        raise ValueError(
            "Article IDs do not match embedding array."
        )

    # Check only the embeddings used by the dataset.
    # Process in chunks to limit temporary memory.
    ids = df["id"].to_numpy()

    for start in range(0, len(ids), 10000):
        chunk_ids = ids[start:start + 10000]

        if not np.isfinite(embeddings[chunk_ids]).all():
            raise ValueError(
                f"Invalid embedding values near article row {start}."
            )

    df = df.merge(
        topics[["id", "kmeans_cluster"]],
        on="id",
        how="left",
        validate="one_to_one",
    )

    if df["kmeans_cluster"].isna().any():
        raise ValueError(
            "Some articles have no KMeans topic assignment."
        )

    df["kmeans_cluster"] = df["kmeans_cluster"].astype(int)

    df = df.sort_values(
        ["date", "id"]
    ).reset_index(drop=True)

    # Use actual calendar month order.
    df["window"] = df["date"].dt.to_period("M").astype(str)

    print(f"Articles loaded: {len(df):,}")
    print(f"Embedding dimensions: {embeddings.shape[1]}")
    print(f"Monthly windows: {df['window'].nunique()}")

    return df, embeddings


# ---------------------------------------------------------
# 2. CATEGORY SHARE ANALYSIS
# ---------------------------------------------------------

def calculate_category_shares(df):

    counts = pd.crosstab(
        df["window"],
        df["category"],
    )

    # Include all observed categories in every month.
    counts = counts.reindex(
        columns=sorted(df["category"].unique()),
        fill_value=0,
    )

    totals = counts.sum(axis=1)

    shares = counts.div(
        totals.replace(0, np.nan),
        axis=0,
    )

    count_long = (
        counts
        .rename_axis(index="window", columns="category")
        .stack(future_stack=True)
        .rename("count")
        .reset_index()
    )

    share_long = (
        shares
        .rename_axis(index="window", columns="category")
        .stack(future_stack=True)
        .rename("share")
        .reset_index()
    )

    result = count_long.merge(
        share_long,
        on=["window", "category"],
        validate="one_to_one",
    )

    result["total"] = result["window"].map(totals)

    return result[
        ["window", "category", "count", "total", "share"]
    ]


# ---------------------------------------------------------
# 3. CATEGORY SHIFT DETECTION
# ---------------------------------------------------------

def detect_category_shifts(
    df,
    min_share_change=0.10,
    alpha=0.05,
):

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    if not 0 <= min_share_change <= 1:
        raise ValueError(
            "min_share_change must be between 0 and 1."
        )

    windows = sorted(df["window"].unique())
    categories = sorted(df["category"].unique())

    records = []

    for previous_window, current_window in zip(
        windows[:-1],
        windows[1:],
    ):

        previous = df[df["window"] == previous_window]
        current = df[df["window"] == current_window]

        previous_total = len(previous)
        current_total = len(current)

        if previous_total == 0 or current_total == 0:
            continue

        p_values = []
        comparison_records = []

        for category in categories:

            previous_count = int(
                (previous["category"] == category).sum()
            )

            current_count = int(
                (current["category"] == category).sum()
            )

            previous_share = previous_count / previous_total
            current_share = current_count / current_total

            change = current_share - previous_share

            try:
                _, p_value = proportions_ztest(
                    count=[
                        previous_count,
                        current_count,
                    ],
                    nobs=[
                        previous_total,
                        current_total,
                    ],
                    alternative="two-sided",
                )

                p_value = float(p_value)

                if not np.isfinite(p_value):
                    p_value = 1.0

            except (ValueError, ZeroDivisionError):
                p_value = 1.0

            p_values.append(p_value)

            comparison_records.append({
                "previous_window": previous_window,
                "window": current_window,
                "category": category,
                "previous_count": previous_count,
                "current_count": current_count,
                "previous_total": previous_total,
                "current_total": current_total,
                "previous_share": previous_share,
                "current_share": current_share,
                "share_change": change,
                "absolute_share_change": abs(change),
                "direction": (
                    "increase" if change > 0
                    else "decrease" if change < 0
                    else "unchanged"
                ),
                "p_value": p_value,
            })

        reject, adjusted_p_values, _, _ = multipletests(
            p_values,
            alpha=alpha,
            method="fdr_bh",
        )

        for record, adjusted_p, significant in zip(
            comparison_records,
            adjusted_p_values,
            reject,
        ):

            record["adjusted_p_value"] = float(adjusted_p)

            record["detected"] = bool(
                significant
                and record["absolute_share_change"]
                >= min_share_change
            )

            records.append(record)

    return pd.DataFrame(records)


# ---------------------------------------------------------
# 4. SEMANTIC DRIFT
# ---------------------------------------------------------

def _sample_embedding_ids(
    df,
    embeddings,
    sample_size,
    rng,
):

    ids = df["id"].to_numpy(dtype=int)

    if len(ids) > sample_size:
        ids = rng.choice(
            ids,
            size=sample_size,
            replace=False,
        )

    return np.asarray(
        embeddings[ids],
        dtype=np.float32,
    )


def calculate_semantic_drift(
    df,
    embeddings,
    sample_size=5000,
    random_state=42,
    baseline_windows=3,
):

    if sample_size < 10:
        raise ValueError("sample_size must be at least 10.")

    windows = sorted(df["window"].unique())

    if len(windows) < 2:
        return pd.DataFrame(columns=[
            "previous_window",
            "window",
            "semantic_drift",
            "max_dimension_drift",
        ])

    baseline_windows = max(
        1,
        min(baseline_windows, len(windows) - 1),
    )

    rng = np.random.default_rng(random_state)

    baseline_df = df[
        df["window"].isin(windows[:baseline_windows])
    ]

    baseline_embeddings = _sample_embedding_ids(
        baseline_df,
        embeddings,
        sample_size,
        rng,
    )

    reference_mean = baseline_embeddings.mean(axis=0)
    reference_std = baseline_embeddings.std(axis=0)

    reference_std = np.where(
        reference_std < 1e-6,
        1.0,
        reference_std,
    )

    bin_edges = np.linspace(-4.0, 4.0, 21)

    monthly_histograms = {}

    for window in windows:

        month_df = df[df["window"] == window]

        month_embeddings = _sample_embedding_ids(
            month_df,
            embeddings,
            sample_size,
            rng,
        )

        standardized = (
            month_embeddings - reference_mean
        ) / reference_std

        standardized = np.clip(
            standardized,
            -4.0,
            4.0,
        )

        histograms = []

        for dimension in range(standardized.shape[1]):

            hist, _ = np.histogram(
                standardized[:, dimension],
                bins=bin_edges,
            )

            hist = hist.astype(np.float64)
            hist += 1e-8
            hist /= hist.sum()

            histograms.append(hist)

        monthly_histograms[window] = np.asarray(histograms)

    records = []

    for previous_window, current_window in zip(
        windows[:-1],
        windows[1:],
    ):

        previous = monthly_histograms[previous_window]
        current = monthly_histograms[current_window]

        distances = np.array([
            jensenshannon(
                previous[dimension],
                current[dimension],
            )
            for dimension in range(previous.shape[0])
        ])

        records.append({
            "previous_window": previous_window,
            "window": current_window,
            "semantic_drift": float(distances.mean()),
            "max_dimension_drift": float(distances.max()),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------
# 5. TOPIC PREVALENCE
# ---------------------------------------------------------

def calculate_topic_prevalence(df):

    counts = pd.crosstab(
        df["window"],
        df["kmeans_cluster"],
    )

    counts = counts.reindex(
        columns=sorted(df["kmeans_cluster"].unique()),
        fill_value=0,
    )

    totals = counts.sum(axis=1)

    prevalence = counts.div(
        totals.replace(0, np.nan),
        axis=0,
    )

    count_long = (
        counts
        .rename_axis(
            index="window",
            columns="kmeans_cluster",
        )
        .stack(future_stack=True)
        .rename("count")
        .reset_index()
    )

    prevalence_long = (
        prevalence
        .rename_axis(
            index="window",
            columns="kmeans_cluster",
        )
        .stack(future_stack=True)
        .rename("prevalence")
        .reset_index()
    )

    result = count_long.merge(
        prevalence_long,
        on=["window", "kmeans_cluster"],
        validate="one_to_one",
    )

    result["total"] = result["window"].map(totals)

    return result[
        [
            "window",
            "kmeans_cluster",
            "count",
            "total",
            "prevalence",
        ]
    ]


# ---------------------------------------------------------
# 6. GROUND TRUTH
# ---------------------------------------------------------

def load_ground_truth(path=GROUND_TRUTH_PATH):

    path = Path(path)

    if not path.exists():
        print(f"Ground truth not found: {path}")
        return None

    records = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not records:
        print("Ground truth file is empty.")
        return None

    for record in records:

        required = {
            "category",
            "window",
            "factor",
        }

        if not required.issubset(record):
            raise ValueError(
                "Ground truth records must contain "
                "category, window and factor."
            )

    return records


# ---------------------------------------------------------
# 7. SYNTHETIC EVALUATION
# ---------------------------------------------------------

def evaluate_detection(
    category_drift,
    ground_truth,
):

    if ground_truth is None:
        return None

    predicted = {
        (row.category, row.window)
        for row in category_drift.itertuples()
        if row.detected
    }

    actual = {
        (record["category"], record["window"])
        for record in ground_truth
    }

    true_positive = len(predicted & actual)
    false_positive = len(predicted - actual)
    false_negative = len(actual - predicted)

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive > 0
        else 0.0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "ground_truth_events": len(actual),
        "predicted_events": len(predicted),
    }


def run_injected_evaluation(
    df,
    ground_truth,
    min_share_change,
    alpha,
    injection_seed=0,
):

    if ground_truth is None:
        print("No ground truth available. Skipping evaluation.")
        return None, None

    from src.stream import inject_trends

    trends = [
        (
            record["category"],
            [record["window"]],
            float(record["factor"]),
        )
        for record in ground_truth
    ]

    injected_df, reconstructed_truth = inject_trends(
        df,
        trends,
        seed=injection_seed,
    )

    # Check reconstruction against saved shares.
    saved_shares = {
        (r["category"], r["window"]): r["share_after"]
        for r in ground_truth
        if "share_after" in r
    }

    for record in reconstructed_truth:

        key = (record["category"], record["window"])

        if key in saved_shares:
            if not np.isclose(
                record["share_after"],
                saved_shares[key],
                atol=0.01,
            ):
                raise ValueError(
                    "Reconstructed injection differs from saved "
                    "ground truth. The original dataset, injection "
                    "seed, or injection order may have changed. "
                    "Do not trust the synthetic evaluation."
                )

    injected_drift = detect_category_shifts(
        injected_df,
        min_share_change=min_share_change,
        alpha=alpha,
    )

    metrics = evaluate_detection(
        injected_drift,
        ground_truth,
    )

    return injected_drift, metrics


# ---------------------------------------------------------
# 8. MAIN PIPELINE
# ---------------------------------------------------------

def run(
    min_share_change=0.10,
    alpha=0.05,
    semantic_sample_size=5000,
    baseline_windows=3,
    evaluate_injected=False,
    injection_seed=0,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df, embeddings = load_data()

    print("\nCalculating category shares...")

    category_shares = calculate_category_shares(df)

    print("Detecting category shifts...")

    category_drift = detect_category_shifts(
        df,
        min_share_change=min_share_change,
        alpha=alpha,
    )

    print("Calculating semantic drift...")

    semantic_drift = calculate_semantic_drift(
        df,
        embeddings,
        sample_size=semantic_sample_size,
        baseline_windows=baseline_windows,
    )

    print("Calculating topic prevalence...")

    topic_prevalence = calculate_topic_prevalence(df)

    # Save historical analysis.
    category_shares.to_csv(
        OUTPUT_DIR / "category_shares.csv",
        index=False,
    )

    category_drift.to_csv(
        OUTPUT_DIR / "category_drift.csv",
        index=False,
    )

    semantic_drift.to_csv(
        OUTPUT_DIR / "semantic_drift.csv",
        index=False,
    )

    topic_prevalence.to_csv(
        OUTPUT_DIR / "topic_prevalence.csv",
        index=False,
    )

    # Optional synthetic evaluation.
    if evaluate_injected:

        print("\nEvaluating injected trends...")

        ground_truth = load_ground_truth()

        injected_drift, metrics = run_injected_evaluation(
            df,
            ground_truth,
            min_share_change,
            alpha,
            injection_seed,
        )

        if injected_drift is not None:

            injected_drift.to_csv(
                OUTPUT_DIR / "category_drift_injected.csv",
                index=False,
            )

        if metrics is not None:

            with open(
                OUTPUT_DIR / "detection_metrics.json",
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    metrics,
                    file,
                    indent=4,
                )

            print("\nSynthetic detection evaluation:")
            print(json.dumps(metrics, indent=4))

    detected_count = (
        int(category_drift["detected"].sum())
        if not category_drift.empty
        else 0
    )

    print("\n" + "=" * 55)
    print("TEMPORAL ANALYSIS COMPLETED")
    print("=" * 55)

    print(f"Articles analysed: {len(df):,}")
    print(f"Monthly windows: {df['window'].nunique()}")
    print(f"Category changes detected: {detected_count}")

    if not semantic_drift.empty:
        print(
            "Average semantic drift:",
            round(
                float(semantic_drift["semantic_drift"].mean()),
                4,
            ),
        )

    print(f"Results saved to: {OUTPUT_DIR}")


# ---------------------------------------------------------
# 9. COMMAND LINE
# ---------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Temporal trend and semantic drift analysis."
    )

    parser.add_argument(
        "--min-share-change",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--semantic-sample-size",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--baseline-windows",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--evaluate-injected",
        action="store_true",
    )

    parser.add_argument(
        "--injection-seed",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    run(
        min_share_change=args.min_share_change,
        alpha=args.alpha,
        semantic_sample_size=args.semantic_sample_size,
        baseline_windows=args.baseline_windows,
        evaluate_injected=args.evaluate_injected,
        injection_seed=args.injection_seed,
    )
