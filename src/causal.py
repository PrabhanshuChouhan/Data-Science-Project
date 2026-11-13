
"""
Event impact analysis.

Examines whether known events are associated with changes
in news category prevalence.

Method:
    - Load monthly category proportions.
    - Compare pre-event and post-event periods.
    - Estimate changes using interrupted time-series regression.
    - Use HAC/Newey-West standard errors.
    - Apply Benjamini-Hochberg correction across tests.

Interpretation:
    Results show temporal associations, not proven causation.

Input:
    results/trends/category_shares.csv
    data/processed/events.csv

Output:
    results/causal/event_impact.csv
    results/causal/event_summary.csv
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


BASE_DIR = Path(__file__).resolve().parent.parent

SHARES_PATH = BASE_DIR / "results" / "trends" / "category_shares.csv"
EVENTS_PATH = BASE_DIR / "data" / "processed" / "events.csv"

OUTPUT_DIR = BASE_DIR / "results" / "causal"


def load_category_shares():
    if not SHARES_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SHARES_PATH}. Run python -m src.trends first."
        )

    df = pd.read_csv(SHARES_PATH)

    # Normalize column names.
    df.columns = [
        str(col).strip().lower().replace(" ", "_")
        for col in df.columns
    ]

    # Find the month/date column, including common alternatives.
    date_candidates = [
        "month", "date", "window", "period", "timestamp",
        "time", "unnamed:_0"
    ]

    date_column = next(
        (col for col in date_candidates if col in df.columns),
        None
    )

    # Sometimes the month is saved as the CSV index.
    if date_column is None and len(df.columns) > 0:
        first_col = df.columns[0]
        parsed = pd.to_datetime(df[first_col], errors="coerce")

        if parsed.notna().mean() >= 0.8:
            date_column = first_col

    if date_column is None:
        raise ValueError(
            "Could not identify the month column in category_shares.csv.\n"
            f"Available columns: {df.columns.tolist()}\n"
            f"First rows:\n{df.head().to_string(index=False)}"
        )

    df[date_column] = pd.to_datetime(
        df[date_column],
        errors="coerce"
    )

    df = df.dropna(subset=[date_column])
    df = df.rename(columns={date_column: "month"})

    # Long format: month, category, share.
    if {"category", "share"}.issubset(df.columns):
        df["category"] = df["category"].astype(str)

        shares = df.pivot_table(
            index="month",
            columns="category",
            values="share",
            aggfunc="mean"
        )

    # Wide format: month, POLITICS, WELLNESS, ...
    else:
        shares = df.set_index("month")
        shares = shares.select_dtypes(include=np.number)

    shares = shares.sort_index()
    shares = shares.replace([np.inf, -np.inf], np.nan)

    if shares.empty:
        raise ValueError(
            "No numeric category-share data found in category_shares.csv."
        )

    print("Loaded category shares:")
    print(f"Months: {len(shares)}")
    print(f"Categories: {shares.columns.tolist()}")
    print(f"Date range: {shares.index.min()} to {shares.index.max()}")

    return shares



def load_events():
    if not EVENTS_PATH.exists():
        template = pd.DataFrame({
            "event": ["Example event - replace this"],
            "date": ["2015-01-01"],
            "category": ["POLITICS"]
        })

        EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        template.to_csv(EVENTS_PATH, index=False)

        raise FileNotFoundError(
            f"Created an event template at {EVENTS_PATH}.\n"
            "Replace the example with real, documented events and rerun."
        )

    events = pd.read_csv(EVENTS_PATH)

    required = {"event", "date", "category"}
    missing = required - set(events.columns)

    if missing:
        raise ValueError(f"events.csv is missing columns: {sorted(missing)}")

    events["date"] = pd.to_datetime(events["date"], errors="coerce")
    events["category"] = events["category"].astype(str).str.strip()

    events = events.dropna(subset=["date"])
    events = events[events["category"] != ""]

    if events.empty:
        raise ValueError("No valid events found in events.csv.")

    events["month"] = events["date"].dt.to_period("M").dt.to_timestamp()

    return events


def analyze_event(
    shares,
    event,
    pre_months=6,
    post_months=6,
    min_periods=3
):
    category = event["category"]

    if category not in shares.columns:
        return None

    event_month = event["month"]

    data = shares[[category]].dropna().copy()
    data = data.rename(columns={category: "share"})

    data["time"] = np.arange(len(data))

    data["post"] = (data.index >= event_month).astype(int)

    event_position = np.where(data.index == event_month)[0]

    if len(event_position) == 0:
        return None

    event_position = int(event_position[0])

    # Restrict analysis to the specified event window.
    start = max(0, event_position - pre_months)
    end = min(len(data), event_position + post_months)

    window = data.iloc[start:end].copy()

    pre = window[window.index < event_month]
    post = window[window.index >= event_month]

    if len(pre) < min_periods or len(post) < min_periods:
        return None

    # Interrupted time-series variables.
    window["time_from_event"] = np.arange(len(window))
    window["post"] = (window.index >= event_month).astype(int)

    first_post_position = np.where(window["post"].to_numpy() == 1)[0][0]
    window["post_time"] = np.maximum(
        0,
        np.arange(len(window)) - first_post_position
    )

    X = window[["time_from_event", "post", "post_time"]]
    X = sm.add_constant(X)

    model = sm.OLS(window["share"], X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": min(3, max(1, len(window) // 4))}
    )

    return {
        "event": event["event"],
        "event_date": event["date"].date().isoformat(),
        "event_month": event_month.strftime("%Y-%m"),
        "category": category,
        "pre_mean": float(pre["share"].mean()),
        "post_mean": float(post["share"].mean()),
        "mean_change": float(post["share"].mean() - pre["share"].mean()),
        "pre_observations": len(pre),
        "post_observations": len(post),
        "immediate_level_change": float(model.params["post"]),
        "level_change_pvalue": float(model.pvalues["post"]),
        "trend_change": float(model.params["post_time"]),
        "trend_change_pvalue": float(model.pvalues["post_time"]),
        "r_squared": float(model.rsquared),
    }


def run_analysis(pre_months=6, post_months=6):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    shares = load_category_shares()
    events = load_events()

    results = []

    for _, event in events.iterrows():
        result = analyze_event(
            shares,
            event,
            pre_months=pre_months,
            post_months=post_months
        )

        if result is not None:
            results.append(result)

    if not results:
        raise ValueError(
            "No event had enough valid pre-event and post-event observations. "
            "Check event dates, categories, and analysis windows."
        )

    results = pd.DataFrame(results)

    # Correct for multiple event/category tests.
    for p_column, q_column in [
        ("level_change_pvalue", "level_change_qvalue"),
        ("trend_change_pvalue", "trend_change_qvalue")
    ]:
        results[q_column] = multipletests(
            results[p_column].fillna(1.0),
            method="fdr_bh"
        )[1]

    results["level_change_significant"] = (
        results["level_change_qvalue"] < 0.05
    )

    results["trend_change_significant"] = (
        results["trend_change_qvalue"] < 0.05
    )

    results["interpretation"] = np.select(
        [
            results["level_change_significant"]
            & (results["mean_change"] > 0),

            results["level_change_significant"]
            & (results["mean_change"] < 0),

            results["trend_change_significant"]
        ],
        [
            "Significant increase associated with event timing",
            "Significant decrease associated with event timing",
            "Significant change in post-event trend"
        ],
        default="No statistically significant change detected"
    )

    results.to_csv(
        OUTPUT_DIR / "event_impact.csv",
        index=False
    )

    summary = (
        results.groupby("category")
        .agg(
            events_analyzed=("event", "count"),
            significant_level_changes=("level_change_significant", "sum"),
            significant_trend_changes=("trend_change_significant", "sum"),
            average_share_change=("mean_change", "mean")
        )
        .reset_index()
    )

    summary.to_csv(
        OUTPUT_DIR / "event_summary.csv",
        index=False
    )

    print("\nEVENT IMPACT ANALYSIS")
    print("=" * 60)
    print(f"Events analyzed: {len(results)}")
    print(f"Significant level changes: {results['level_change_significant'].sum()}")
    print(f"Significant trend changes: {results['trend_change_significant'].sum()}")
    print(f"\nSaved results to: {OUTPUT_DIR}")

    print("\nEvent results:")
    print(
        results[
            [
                "event",
                "category",
                "mean_change",
                "level_change_qvalue",
                "interpretation"
            ]
        ].to_string(index=False)
    )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze news category changes around known events."
    )

    parser.add_argument("--pre-months", type=int, default=6)
    parser.add_argument("--post-months", type=int, default=6)

    args = parser.parse_args()

    if args.pre_months < 3 or args.post_months < 3:
        parser.error("Use at least 3 months for both pre and post windows.")

    run_analysis(args.pre_months, args.post_months)


if __name__ == "__main__":
    main()
