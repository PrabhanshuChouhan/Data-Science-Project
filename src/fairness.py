
"""
Representation and fairness audit for the news dataset.

Analyses:
    1. Missing author/byline coverage.
    2. Article concentration among bylines.
    3. Category representation across author-activity cohorts.
    4. Category-share disparities across cohorts.
    5. Category distribution divergence across cohorts.

Important:
    The dataset does not provide reliable demographic labels.
    This module does not infer protected attributes from names.

    Author-activity cohorts are descriptive proxies, not
    protected demographic groups or proof of discrimination.

Input:
    data/processed/news.parquet

Outputs:
    results/fairness/author_concentration.csv
    results/fairness/category_by_author_cohort.csv
    results/fairness/category_disparity.csv
    results/fairness/fairness_summary.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "news.parquet"

OUTPUT_DIR = BASE_DIR / "results" / "fairness"


# ---------------------------------------------------------
# Load and validate data
# ---------------------------------------------------------

def load_data(data_path=DATA_PATH):
    """Load article data and validate required columns."""

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    df = pd.read_parquet(data_path)

    required = {"category", "authors", "date"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    df = df[["category", "authors", "date"]].copy()

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["category"] = (
        df["category"]
        .astype("string")
        .str.strip()
    )

    # Normalize missing and blank bylines.
    df["authors"] = (
        df["authors"]
        .astype("string")
        .str.strip()
    )

    df["authors"] = df["authors"].replace(
        ["", "nan", "None", "null"],
        pd.NA
    )

    df = df.dropna(
        subset=["category", "date"]
    )

    if df.empty:
        raise ValueError("No valid articles found.")

    df["author_status"] = np.where(
        df["authors"].isna(),
        "Unknown",
        "Known"
    )

    # Treat the full byline string as one unit.
    # Do not assume comma-separated names are always
    # separate individual authors.
    df["byline_group"] = df["authors"].fillna(
        "UNKNOWN_BYLINE"
    )

    return df


# ---------------------------------------------------------
# Author coverage
# ---------------------------------------------------------

def calculate_author_coverage(df):
    """Measure the completeness of author/byline metadata."""

    total = len(df)

    known = int(
        (df["author_status"] == "Known").sum()
    )

    unknown = total - known

    return {
        "total_articles": total,
        "articles_with_known_byline": known,
        "articles_with_unknown_byline": unknown,
        "known_byline_percentage": float(
            known / total * 100
        ),
        "unknown_byline_percentage": float(
            unknown / total * 100
        ),
        "unique_known_bylines": int(
            df.loc[
                df["author_status"] == "Known",
                "byline_group"
            ].nunique()
        )
    }


# ---------------------------------------------------------
# Author concentration
# ---------------------------------------------------------

def calculate_author_concentration(df):
    """
    Measure how concentrated articles are among known bylines.

    Metrics:
        - Article count per byline
        - Share of articles from top 1%, 5%, and 10% of bylines
        - Herfindahl-Hirschman Index (HHI)
        - Gini coefficient

    Unknown bylines are excluded from author-level concentration
    calculations but reported separately in coverage statistics.
    """

    known = df[
        df["author_status"] == "Known"
    ]

    counts = (
        known.groupby("byline_group")
        .size()
        .rename("article_count")
        .reset_index()
        .sort_values(
            "article_count",
            ascending=False
        )
    )

    if counts.empty:
        return counts, {
            "known_byline_articles": 0,
            "unique_bylines": 0,
            "hhi": None,
            "gini": None,
            "top_1_percent_article_share": None,
            "top_5_percent_article_share": None,
            "top_10_percent_article_share": None
        }

    total_articles = counts["article_count"].sum()
    total_bylines = len(counts)

    shares = (
        counts["article_count"].to_numpy(dtype=float)
        / total_articles
    )

    sorted_counts = np.sort(
        counts["article_count"].to_numpy(dtype=float)
    )

    n = len(sorted_counts)

    # Standard Gini coefficient.
    gini = (
        2 * np.sum(
            np.arange(1, n + 1) * sorted_counts
        )
        / (n * sorted_counts.sum())
        - (n + 1) / n
    )

    def top_share(percent):
        number = max(
            1,
            int(np.ceil(total_bylines * percent / 100))
        )

        return float(
            counts.head(number)["article_count"].sum()
            / total_articles
        )

    summary = {
        "known_byline_articles": int(total_articles),
        "unique_bylines": int(total_bylines),
        "hhi": float(np.sum(shares ** 2)),
        "gini": float(gini),
        "top_1_percent_article_share": top_share(1),
        "top_5_percent_article_share": top_share(5),
        "top_10_percent_article_share": top_share(10)
    }

    return counts, summary


# ---------------------------------------------------------
# Author activity cohorts
# ---------------------------------------------------------

def assign_author_cohorts(df):
    """
    Assign known bylines to low, medium, or high activity cohorts.

    Cohorts are determined by the distribution of article
    counts per byline.

    This is a descriptive grouping, not a protected attribute.
    """

    result = df.copy()

    known = result[
        result["author_status"] == "Known"
    ]

    author_counts = (
        known.groupby("byline_group")
        .size()
        .rename("author_article_count")
    )

    if author_counts.empty:
        result["author_cohort"] = "Unknown"
        return result

    # Rank-based tertiles avoid qcut failures when many
    # bylines have the same number of articles.
    ranked = author_counts.rank(
        method="first",
        pct=True
    )

    cohort_map = pd.cut(
        ranked,
        bins=[0, 1 / 3, 2 / 3, 1.0],
        labels=["Low activity", "Medium activity", "High activity"],
        include_lowest=True
    )

    cohort_map = cohort_map.astype("string")

    result["author_cohort"] = (
        result["byline_group"]
        .map(cohort_map)
        .fillna("Unknown")
    )

    result["author_article_count"] = (
        result["byline_group"]
        .map(author_counts)
        .fillna(0)
        .astype(int)
    )

    return result


# ---------------------------------------------------------
# Category representation by cohort
# ---------------------------------------------------------

def calculate_category_representation(df):
    """
    Calculate category counts and shares across author cohorts.

    Unknown bylines are kept as a separate cohort.
    """

    counts = (
        df.groupby(
            ["author_cohort", "category"],
            observed=True
        )
        .size()
        .rename("article_count")
        .reset_index()
    )

    totals = (
        counts.groupby("author_cohort")["article_count"]
        .sum()
        .rename("cohort_total")
        .reset_index()
    )

    result = counts.merge(
        totals,
        on="author_cohort",
        validate="many_to_one"
    )

    result["category_share"] = (
        result["article_count"]
        / result["cohort_total"]
    )

    return result


# ---------------------------------------------------------
# Category disparity
# ---------------------------------------------------------

def calculate_category_disparity(representation):
    """
    Compare category shares across author-activity cohorts.

    For each category:
        - Minimum cohort share
        - Maximum cohort share
        - Absolute share gap

    This is descriptive disparity, not evidence of bias.
    """

    pivot = representation.pivot_table(
        index="category",
        columns="author_cohort",
        values="category_share",
        fill_value=0,
        observed=True
    )

    cohort_columns = [
        col for col in pivot.columns
        if col != "Unknown"
    ]

    if not cohort_columns:
        raise ValueError(
            "No known author-activity cohorts available."
        )

    result = pd.DataFrame({
        "category": pivot.index,
        "minimum_cohort_share": (
            pivot[cohort_columns].min(axis=1).to_numpy()
        ),
        "maximum_cohort_share": (
            pivot[cohort_columns].max(axis=1).to_numpy()
        ),
        "absolute_share_gap": (
            pivot[cohort_columns].max(axis=1)
            - pivot[cohort_columns].min(axis=1)
        ).to_numpy()
    })

    result = result.sort_values(
        "absolute_share_gap",
        ascending=False
    )

    return result, pivot


# ---------------------------------------------------------
# Distribution divergence
# ---------------------------------------------------------

def calculate_distribution_divergence(pivot):
    """
    Calculate Jensen-Shannon distance between each cohort's
    category distribution and the overall known-byline distribution.

    Lower values mean more similar category distributions.

    This measures distributional difference, not fairness itself.
    """

    known_columns = [
        col for col in pivot.columns
        if col != "Unknown"
    ]

    if not known_columns:
        return pd.DataFrame(
            columns=[
                "cohort",
                "jensen_shannon_distance"
            ]
        )

    distributions = pivot[known_columns].to_numpy(
        dtype=float
    )

    # Each column represents a cohort and each row a category.
    cohort_totals = distributions.sum(axis=0)

    valid = cohort_totals > 0

    distributions = distributions[:, valid]
    valid_columns = np.array(known_columns)[valid]

    if distributions.shape[1] == 0:
        return pd.DataFrame(
            columns=[
                "cohort",
                "jensen_shannon_distance"
            ]
        )

    distributions = (
        distributions
        / distributions.sum(axis=0, keepdims=True)
    )

    overall = (
        distributions * cohort_totals[valid][None, :]
    ).sum(axis=1)

    overall = overall / overall.sum()

    records = []

    for index, cohort in enumerate(valid_columns):
        distance = jensenshannon(
            distributions[:, index],
            overall,
            base=2
        )

        records.append({
            "cohort": str(cohort),
            "jensen_shannon_distance": float(distance)
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------

def run(data_path=DATA_PATH):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading article data...")

    df = load_data(data_path)

    print(f"Articles: {len(df):,}")
    print(f"Categories: {df['category'].nunique()}")

    # 1. Coverage
    coverage = calculate_author_coverage(df)

    # 2. Author concentration
    print("\nCalculating author concentration...")

    author_counts, concentration = (
        calculate_author_concentration(df)
    )

    author_counts.to_csv(
        OUTPUT_DIR / "author_concentration.csv",
        index=False
    )

    # 3. Assign activity cohorts
    print("Assigning author-activity cohorts...")

    df = assign_author_cohorts(df)

    # 4. Category representation
    print("Calculating category representation...")

    representation = calculate_category_representation(df)

    representation.to_csv(
        OUTPUT_DIR / "category_by_author_cohort.csv",
        index=False
    )

    # 5. Category disparity
    disparity, pivot = calculate_category_disparity(
        representation
    )

    disparity.to_csv(
        OUTPUT_DIR / "category_disparity.csv",
        index=False
    )

    # 6. Distribution divergence
    divergence = calculate_distribution_divergence(
        pivot
    )

    divergence.to_csv(
        OUTPUT_DIR / "cohort_distribution_divergence.csv",
        index=False
    )

    # 7. Cohort sizes
    cohort_sizes = (
        df.groupby("author_cohort", observed=True)
        .size()
        .to_dict()
    )

    summary = {
        "scope": "Descriptive authorship and representation audit",
        "total_articles": int(len(df)),
        "number_of_categories": int(df["category"].nunique()),
        "author_coverage": coverage,
        "author_concentration": concentration,
        "cohort_article_counts": {
            str(key): int(value)
            for key, value in cohort_sizes.items()
        },
        "maximum_category_share_gap": (
            float(disparity["absolute_share_gap"].max())
            if not disparity.empty else None
        ),
        "limitations": [
            (
                "No reliable demographic or protected-group "
                "attributes are available."
            ),
            (
                "Author-activity cohorts are descriptive proxies, "
                "not protected groups."
            ),
            (
                "A byline string may represent multiple authors "
                "or inconsistent author formatting."
            ),
            (
                "Category-share differences do not establish "
                "discrimination or its cause."
            ),
            (
                "The analysis measures representation in this "
                "dataset, not the full news ecosystem."
            )
        ]
    }

    with open(
        OUTPUT_DIR / "fairness_summary.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            summary,
            file,
            indent=4
        )

    # -----------------------------------------------------
    # Report
    # -----------------------------------------------------

    print("\nREPRESENTATION AUDIT")
    print("=" * 55)

    print(
        f"Known byline coverage: "
        f"{coverage['known_byline_percentage']:.2f}%"
    )

    print(
        f"Unknown byline coverage: "
        f"{coverage['unknown_byline_percentage']:.2f}%"
    )

    print(
        f"Unique known bylines: "
        f"{coverage['unique_known_bylines']:,}"
    )

    print(
        f"Author concentration (Gini): "
        f"{concentration['gini']}"
    )

    print(
        f"Author concentration (HHI): "
        f"{concentration['hhi']}"
    )

    print("\nCategory share disparities:")

    print(
        disparity.to_string(index=False)
    )

    print("\nCohort distribution divergence:")

    print(
        divergence.to_string(index=False)
    )

    print("\nSaved outputs to:")
    print(OUTPUT_DIR)


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Audit authorship concentration and "
            "category representation."
        )
    )

    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_PATH
    )

    args = parser.parse_args()

    run(
        data_path=args.data_path
    )
