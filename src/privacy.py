
"""
Differentially private monthly category analytics.

Mechanism:
    IBM diffprivlib Laplace histogram.

Privacy unit:
    One article record.

Adjacency:
    Add/remove one article record.

Each article contributes to exactly one category bin
in exactly one monthly histogram.

Privacy accounting:
    - Sequential composition: epsilon * number of months.
    - Parallel composition: epsilon, because monthly datasets
      are disjoint under article-level adjacency.

Important:
    Parallel composition does NOT apply if the privacy unit
    is a person/author who may contribute multiple articles
    across different months.

Outputs:
    results/privacy/private_category_statistics.csv
    results/privacy/privacy_metadata.json
    results/privacy/internal/utility_evaluation.json

The internal utility file contains exact-data comparisons.
Never publish it as part of the private release.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from diffprivlib.tools import histogram


# ---------------------------------------------------------
# Paths and public metadata
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "news.parquet"

OUTPUT_DIR = BASE_DIR / "results" / "privacy"
INTERNAL_DIR = OUTPUT_DIR / "internal"

# Fixed category universe.
# These category labels are treated as public metadata.
PUBLIC_CATEGORIES = [
    "ENTERTAINMENT",
    "FOOD & DRINK",
    "PARENTING",
    "POLITICS",
    "QUEER VOICES",
    "STYLE & BEAUTY",
    "TRAVEL",
    "WELLNESS",
]

# Fixed public analysis period.
START_MONTH = "2012-01"
END_MONTH = "2018-12"


# ---------------------------------------------------------
# Load and validate data
# ---------------------------------------------------------

def load_data(data_path=DATA_PATH):
    """Load and validate article data."""

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    df = pd.read_parquet(data_path)

    required = {"date", "category"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df = df[["date", "category"]].copy()

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["category"] = (
        df["category"]
        .astype("string")
        .str.strip()
    )

    df = df.dropna(
        subset=["date", "category"]
    )

    # Keep the predefined analysis period.
    df["month"] = (
        df["date"]
        .dt.to_period("M")
        .astype(str)
    )

    df = df[
        df["month"].between(START_MONTH, END_MONTH)
    ].copy()

    # Do not silently discard unexpected categories.
    unknown = sorted(
        set(df["category"].unique()) - set(PUBLIC_CATEGORIES)
    )

    if unknown:
        raise ValueError(
            "Unexpected categories found in the dataset: "
            f"{unknown}\n"
            "Update PUBLIC_CATEGORIES only after deciding "
            "that the category universe is public."
        )

    if df.empty:
        raise ValueError(
            "No articles found in the configured analysis period."
        )

    return df


# ---------------------------------------------------------
# Differentially private histogram
# ---------------------------------------------------------

def private_histogram(
    category_values,
    categories,
    epsilon,
    random_state=None,
):
    """
    Release a differentially private category histogram.

    Each article contributes to exactly one bin.

    Under add/remove-one-article adjacency, the L1
    sensitivity of this histogram is 1.
    """

    if epsilon <= 0:
        raise ValueError("Epsilon must be greater than zero.")

    category_to_index = {
        category: index
        for index, category in enumerate(categories)
    }

    unknown = set(category_values) - set(category_to_index)

    if unknown:
        raise ValueError(
            f"Unknown categories encountered: {sorted(unknown)}"
        )

    encoded = np.array(
        [
            category_to_index[value]
            for value in category_values
        ],
        dtype=int
    )

    number_of_categories = len(categories)

    # Bins are centered at 0, 1, ..., k-1.
    noisy_counts, _ = histogram(
        encoded,
        bins=number_of_categories,
        range=(-0.5, number_of_categories - 0.5),
        epsilon=epsilon,
        random_state=random_state,
    )

    # Non-negative clipping is privacy-preserving
    # post-processing.
    noisy_counts = np.maximum(
        noisy_counts,
        0
    )

    return noisy_counts


# ---------------------------------------------------------
# Monthly private statistics
# ---------------------------------------------------------

def calculate_private_statistics(
    df,
    epsilon_per_month=1.0,
    random_state=None,
):
    """
    Calculate private monthly category counts and shares.

    The same fixed category universe and monthly calendar
    are used for every release.

    random_state=None uses diffprivlib's default randomness.
    Supplying an integer is intended only for reproducible
    experiments, not production privacy releases.
    """

    if epsilon_per_month <= 0:
        raise ValueError(
            "Epsilon must be greater than zero."
        )

    categories = PUBLIC_CATEGORIES.copy()

    windows = [
        str(month)
        for month in pd.period_range(
            START_MONTH,
            END_MONTH,
            freq="M"
        )
    ]

    records = []

    # Deterministic seeds are generated only when explicitly
    # requested for an experiment.
    rng = (
        np.random.default_rng(random_state)
        if random_state is not None
        else None
    )

    for window in windows:

        monthly = df[
            df["month"] == window
        ]

        values = monthly["category"].tolist()

        release_seed = (
            int(rng.integers(0, 2**31 - 1))
            if rng is not None
            else None
        )

        noisy_counts = private_histogram(
            category_values=values,
            categories=categories,
            epsilon=epsilon_per_month,
            random_state=release_seed,
        )

        noisy_total = float(noisy_counts.sum())

        if noisy_total > 0:
            noisy_shares = noisy_counts / noisy_total
        else:
            noisy_shares = np.zeros(
                len(categories),
                dtype=float
            )

        for category, count, share in zip(
            categories,
            noisy_counts,
            noisy_shares
        ):
            records.append({
                "window": window,
                "category": category,
                "noisy_count": float(count),
                "noisy_share": float(share),
            })

    return (
        pd.DataFrame(records),
        categories,
        windows
    )


# ---------------------------------------------------------
# Exact baseline for INTERNAL utility evaluation
# ---------------------------------------------------------

def calculate_exact_statistics(
    df,
    categories,
    windows
):
    """
    Calculate exact statistics for internal evaluation only.

    This data is NOT differentially private and must not
    be included in a public release.
    """

    counts = (
        df.groupby(["month", "category"])
        .size()
        .rename("exact_count")
        .reset_index()
        .rename(columns={"month": "window"})
    )

    # Create the full month-category grid so that categories
    # with zero articles in a month are represented as zero.
    full_index = pd.MultiIndex.from_product(
        [windows, categories],
        names=["window", "category"]
    )

    counts = (
        counts.set_index(["window", "category"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    totals = (
        counts.groupby("window")["exact_count"]
        .sum()
        .rename("exact_total")
        .reset_index()
    )

    counts = counts.merge(
        totals,
        on="window",
        validate="many_to_one"
    )

    counts["exact_share"] = np.divide(
        counts["exact_count"],
        counts["exact_total"],
        out=np.zeros(len(counts), dtype=float),
        where=counts["exact_total"].to_numpy() > 0
    )

    return counts


# ---------------------------------------------------------
# Internal utility evaluation
# ---------------------------------------------------------

def evaluate_utility(
    private_results,
    exact_results
):
    """
    Compare private and exact estimates.

    These metrics are data-dependent and are NOT a
    differentially private release.
    """

    comparison = private_results.merge(
        exact_results,
        on=["window", "category"],
        how="inner",
        validate="one_to_one"
    )

    if comparison.empty:
        raise ValueError(
            "No records available for utility evaluation."
        )

    share_errors = (
        comparison["noisy_share"]
        - comparison["exact_share"]
    )

    return {
        "share_mae": float(
            np.mean(np.abs(share_errors))
        ),
        "share_rmse": float(
            np.sqrt(np.mean(share_errors ** 2))
        ),
        "evaluated_records": int(len(comparison)),
        "warning": (
            "Internal evaluation only. "
            "These metrics are not differentially private."
        )
    }


# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------

def run(
    data_path=DATA_PATH,
    epsilon_per_month=1.0,
    random_state=None
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    INTERNAL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading article data...")

    df = load_data(data_path)

    windows = [
        str(month)
        for month in pd.period_range(
            START_MONTH,
            END_MONTH,
            freq="M"
        )
    ]

    print(f"Articles: {len(df):,}")
    print(f"Months in public calendar: {len(windows)}")
    print(f"Categories: {len(PUBLIC_CATEGORIES)}")

    print("\nGenerating differentially private histograms...")

    private_results, categories, windows = (
        calculate_private_statistics(
            df,
            epsilon_per_month=epsilon_per_month,
            random_state=random_state
        )
    )

    # Exact baseline stays internal.
    exact_results = calculate_exact_statistics(
        df,
        categories,
        windows
    )

    utility = evaluate_utility(
        private_results,
        exact_results
    )

    # -----------------------------------------------------
    # Save private release
    # -----------------------------------------------------

    private_path = (
        OUTPUT_DIR / "private_category_statistics.csv"
    )

    private_results.to_csv(
        private_path,
        index=False
    )

    # Conservative sequential composition bound.
    sequential_epsilon = (
        epsilon_per_month * len(windows)
    )

    # Parallel composition applies under the stated
    # article-level adjacency and disjoint month partitions.
    parallel_epsilon = epsilon_per_month

    metadata = {
        "mechanism": (
            "IBM diffprivlib Laplace histogram"
        ),
        "epsilon_per_month": epsilon_per_month,
        "delta": 0.0,
        "privacy_unit": "one article record",
        "adjacency": (
            "Add/remove one article. Each article contributes "
            "to exactly one category bin in exactly one month."
        ),
        "number_of_monthly_releases": len(windows),
        "accounting": {
            "sequential_composition_upper_bound": (
                sequential_epsilon
            ),
            "parallel_composition_epsilon": (
                parallel_epsilon
            ),
            "parallel_composition_assumption": (
                "Monthly article datasets are disjoint and "
                "privacy adjacency is at the article-record level."
            )
        },
        "categories": categories,
        "windows": windows,
        "limitations": [
            (
                "This protects article records, not authors "
                "or individuals who may contribute multiple articles."
            ),
            (
                "Parallel composition relies on disjoint monthly "
                "partitions and article-level adjacency."
            ),
            (
                "Category labels and the calendar are treated "
                "as public, fixed metadata."
            ),
            (
                "Article text, embeddings, graph structure, and "
                "other derived outputs are not privatized."
            ),
            (
                "The exact utility evaluation is stored separately "
                "and must not be published."
            ),
            (
                "A fixed random seed is for experiments only. "
                "Use secure randomness for real privacy releases."
            )
        ]
    }

    metadata_path = (
        OUTPUT_DIR / "privacy_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4
        )

    # -----------------------------------------------------
    # Save internal utility evaluation separately
    # -----------------------------------------------------

    utility_path = (
        INTERNAL_DIR / "utility_evaluation.json"
    )

    with open(
        utility_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            utility,
            file,
            indent=4
        )

    # -----------------------------------------------------
    # Report
    # -----------------------------------------------------

    print("\nPrivacy analysis completed.")
    print("-" * 55)

    print(
        f"Epsilon per month: {epsilon_per_month}"
    )

    print(
        f"Number of monthly releases: {len(windows)}"
    )

    print(
        f"Sequential composition upper bound: "
        f"{sequential_epsilon:.4f}"
    )

    print(
        f"Parallel composition epsilon: "
        f"{parallel_epsilon:.4f}"
    )

    print("\nInternal utility evaluation:")
    print(
        json.dumps(
            utility,
            indent=4
        )
    )

    print("\nSaved private release:")
    print(private_path)
    print(metadata_path)

    print("\nSaved internal-only evaluation:")
    print(utility_path)

    print(
        "\nWARNING: Do not publish the internal directory "
        "or its utility evaluation file."
    )


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Differentially private monthly news analytics."
        )
    )

    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_PATH
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0,
        help=(
            "Privacy budget for each monthly histogram. "
            "Default: 1.0."
        )
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=None,
        help=(
            "Optional deterministic seed for experiments only. "
            "Omit for the default randomness."
        )
    )

    args = parser.parse_args()

    run(
        data_path=args.data_path,
        epsilon_per_month=args.epsilon,
        random_state=args.random_state
    )
