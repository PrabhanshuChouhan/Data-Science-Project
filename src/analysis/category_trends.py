from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = (
    BASE_DIR / "results" / "streaming" / "reports"
)

OUTPUT_DIR = (
    BASE_DIR / "results" / "streaming" / "analysis"
)

def load_monthly_articles():
    """Load all monthly topic reports."""

    files = sorted(REPORT_DIR.glob("*_topics.parquet"))
    if not files:
        raise FileNotFoundError(
            "No monthly topic reports found. Run the streaming engine first."
        )
        
    frames=[]
    
    for file in files:
        df = pd.read_parquet(file)
        frames.append(df)
    
    return pd.concat(frames, ignore_index=True)

def calculate_category_trends(df):
    """Calculate monthly category counts and proportions."""
    # Group by month, topic_id, and category to get counts
    data=df.copy()
    
    data["month"] = (pd.to_datetime(data["date"]).dt.to_period("M").astype(str))

    monthly_counts = (
        data.groupby(["month", "category"])
        .size().unstack(fill_value=0)
        .sort_index()
    )

    # Include zero-count months for every category.
    monthly_counts = monthly_counts.reindex(
        sorted(monthly_counts.index),
        fill_value=0,
    )

    monthly_shares = monthly_counts.div(
        monthly_counts.sum(axis=1).replace(0, np.nan),
        axis=0,
    ).fillna(0) * 100
    
    # Difference in percentage points, not relative percentage change.
    share_changes = monthly_shares.diff().fillna(0)

    results = []

    for month in monthly_shares.index:

        for category in monthly_shares.columns:

            current_share = float(
                monthly_shares.loc[month, category]
            )

            previous_share = float(
                monthly_shares.shift(1).loc[month, category]
            ) if month != monthly_shares.index[0] else current_share

            change = current_share - previous_share

            if change > 1.0:
                direction = "rising"
            elif change < -1.0:
                direction = "falling"
            else:
                direction = "stable"

            results.append({
                "month": month,
                "category": str(category),
                "article_count": int(
                    monthly_counts.loc[month, category]
                ),
                "share_percent": round(current_share, 3),
                "previous_share_percent": round(previous_share, 3),
                "change_percentage_points": round(change, 3),
                "direction": direction,
            })

    trend_df = pd.DataFrame(results)

    return monthly_counts, monthly_shares, trend_df

def find_largest_shifts(trend_df,top_n=10):
    """Find the largest category movements in each month."""
     
    shifts=trend_df.copy()
    shifts["absolute_change"] = shifts["change_percentage_points"].abs()
    
    return(
        shifts.sort_values(
            ["month", "absolute_change"],
            ascending=[True, False]
        )
        .groupby("month")
        .head(top_n)
        .drop(columns=["absolute_change"])
        .reset_index(drop=True)
    )
    
def run_category_analysis():
    print("Loading monthly news reports...")

    df = load_monthly_articles()

    print(f"Articles loaded: {len(df)}")

    counts, shares, trends = calculate_category_trends(df)

    largest_shifts = find_largest_shifts(trends)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    counts.to_csv(
        OUTPUT_DIR / "category_monthly_counts.csv"
    )

    shares.to_csv(
        OUTPUT_DIR / "category_monthly_shares.csv"
    )

    trends.to_csv(
        OUTPUT_DIR / "category_trends.csv",
        index=False,
    )

    largest_shifts.to_csv(
        OUTPUT_DIR / "largest_category_shifts.csv",
        index=False,
    )

    print("\nCategory analysis completed.")

    print(f"Months analyzed: {len(counts)}")
    print(f"Categories tracked: {len(counts.columns)}")

    print("\nLatest category trends:")

    latest_month = trends["month"].max()

    latest = trends[
        trends["month"] == latest_month
    ].sort_values(
        "share_percent",
        ascending=False,
    )

    print(latest.to_string(index=False))

    print("\nResults saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    run_category_analysis()
