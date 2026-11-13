
import json
import warnings
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "news.parquet"
RESULTS_DIR = BASE_DIR / "results"
GROUND_TRUTH_PATH = RESULTS_DIR / "stream" / "trend_ground_truth.json"


# ---------------------------------------------------------
# 1. DIVIDE DATA INTO TIME WINDOWS
# ---------------------------------------------------------

def add_window(
    df: pd.DataFrame,
    freq: str = "M",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Assign every article to a chronological time window.

    Supported frequencies:
        M = Monthly
        W = Weekly
        Q = Quarterly

    Returns a copy of the dataframe with a 'window' column.
    """

    if date_col not in df.columns:
        raise KeyError(f"Column '{date_col}' not found.")

    if freq.upper() not in {"M", "W", "Q"}:
        raise ValueError("freq must be 'M', 'W', or 'Q'.")

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")

    if out[date_col].isna().any():
        raise ValueError("Some rows contain invalid or missing dates.")

    periods = out[date_col].dt.to_period(freq.upper())

    if freq.upper() == "W":
        out["window"] = periods.dt.start_time.dt.strftime("%Y-%m-%d")
    else:
        out["window"] = periods.astype(str)

    return out


# ---------------------------------------------------------
# 2. REPORT BATCH SIZES
# ---------------------------------------------------------

def window_report(
    df: pd.DataFrame,
    window_col: str = "window",
    min_batch: int = 300,
) -> pd.DataFrame:
    """
    Report the number of articles in every time window.
    """

    if window_col not in df.columns:
        raise KeyError(f"Column '{window_col}' not found.")

    if min_batch < 1:
        raise ValueError("min_batch must be at least 1.")

    sizes = (
        df.groupby(window_col)
        .size()
        .rename("n_items")
        .to_frame()
        .sort_index()
    )

    if sizes.empty:
        print("No time windows found.")
        sizes["ok"] = pd.Series(dtype=bool)
        return sizes

    sizes["ok"] = sizes["n_items"] >= min_batch

    print(f"Total windows: {len(sizes)}")
    print(f"Minimum batch: {sizes['n_items'].min()}")
    print(f"Median batch: {sizes['n_items'].median():.0f}")
    print(f"Maximum batch: {sizes['n_items'].max()}")
    print(
        f"Windows below {min_batch} articles: "
        f"{(~sizes['ok']).sum()}"
    )

    return sizes


# ---------------------------------------------------------
# 3. GENERATE CHRONOLOGICAL MICRO-BATCHES
# ---------------------------------------------------------

def stream(
    df: pd.DataFrame,
    window_col: str = "window",
    min_batch: int = 0,
    sort_within: str | None = "date",
) -> Iterator[tuple[str, pd.DataFrame]]:
    """
    Yield one time window at a time.

    Each result contains:
        window_label
        batch_dataframe

    Batches are yielded in chronological order.
    """

    if window_col not in df.columns:
        raise KeyError(
            f"'{window_col}' not found. Call add_window() first."
        )

    if min_batch < 0:
        raise ValueError("min_batch cannot be negative.")

    if sort_within is not None and sort_within not in df.columns:
        raise KeyError(f"Sort column '{sort_within}' not found.")

    skipped = []

    for label, batch in df.groupby(window_col, sort=True):

        if len(batch) < min_batch:
            skipped.append((label, len(batch)))
            continue

        if sort_within:
            batch = batch.sort_values(sort_within)

        yield str(label), batch.reset_index(drop=True)

    if skipped:
        warnings.warn(
            f"Skipped {len(skipped)} small windows. "
            f"Examples: {skipped[:5]}",
            stacklevel=2,
        )


# ---------------------------------------------------------
# 4. ARTIFICIAL TREND INJECTION
# ---------------------------------------------------------

def inject_trends(
    df: pd.DataFrame,
    trends: Sequence[tuple[str, Sequence[str], float]],
    window_col: str = "window",
    category_col: str = "category",
    max_share: float = 0.9,
    seed: int = 0,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Artificially increase the share of selected categories.

    Each trend is:
        (category, [windows], factor)

    Example:
        ("POLITICS", ["2014-09", "2014-10"], 2.0)

    The function removes randomly selected articles from
    other categories. It does not duplicate articles.

    Returns:
        modified_dataframe
        ground_truth_records
    """

    for col in (window_col, category_col):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found.")

    if not 0 < max_share <= 1:
        raise ValueError("max_share must be in (0, 1].")

    rng = np.random.default_rng(seed)

    keep = np.ones(len(df), dtype=bool)

    windows = df[window_col].to_numpy()
    categories = df[category_col].to_numpy()

    available_categories = set(categories)
    truth = []

    for category, target_windows, factor in trends:

        if factor <= 1:
            raise ValueError(
                f"Trend factor must be greater than 1: {factor}"
            )

        if category not in available_categories:
            raise ValueError(
                f"Category '{category}' does not exist."
            )

        for window in target_windows:

            in_window = (windows == window) & keep

            if not in_window.any():
                warnings.warn(
                    f"Window '{window}' is empty. Skipping trend."
                )
                continue

            is_target = categories == category

            n_target = int((in_window & is_target).sum())
            n_total = int(in_window.sum())

            if n_target == 0:
                warnings.warn(
                    f"No articles for '{category}' in '{window}'."
                )
                continue

            share_before = n_target / n_total

            share_after_target = min(
                factor * share_before,
                max_share,
            )

            n_other_keep = int(
                round(
                    n_target
                    * (1 - share_after_target)
                    / share_after_target
                )
            )

            other_indices = np.flatnonzero(
                in_window & ~is_target
            )

            if n_other_keep < len(other_indices):

                n_remove = len(other_indices) - n_other_keep

                remove_indices = rng.choice(
                    other_indices,
                    size=n_remove,
                    replace=False,
                )

                keep[remove_indices] = False

            updated_window = (windows == window) & keep

            actual_share = (
                (is_target & updated_window).sum()
                / updated_window.sum()
            )

            truth.append({
                "category": category,
                "window": window,
                "factor": float(factor),
                "share_before": round(float(share_before), 4),
                "share_after": round(float(actual_share), 4),
            })

    modified_df = df.loc[keep].copy().reset_index(drop=True)

    return modified_df, truth


# ---------------------------------------------------------
# 5. SAVE AND LOAD EXPERIMENT GROUND TRUTH
# ---------------------------------------------------------

def save_truth(
    truth: list[dict],
    path: str | Path = GROUND_TRUTH_PATH,
) -> None:
    """
    Save artificial trend information for later evaluation.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(truth, indent=2),
        encoding="utf-8",
    )

    print(f"Ground truth saved: {path}")


def load_truth(
    path: str | Path = GROUND_TRUTH_PATH,
) -> set[tuple[str, str]]:
    """
    Load ground truth as (category, window) pairs.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Ground truth file not found: {path}"
        )

    records = json.loads(path.read_text(encoding="utf-8"))

    return {
        (record["category"], record["window"])
        for record in records
    }


# ---------------------------------------------------------
# 6. MISSING LABEL SIMULATION
# ---------------------------------------------------------

def inject_missing(
    df: pd.DataFrame,
    col: str = "category",
    frac: float = 0.15,
    seed: int = 1,
    mechanism: str = "mcar",
    depends_on: str | None = None,
) -> pd.DataFrame:
    """
    Hide some category labels for missing-data experiments.

    Original column remains unchanged.

    New columns:
        category_obs
        category_missing

    MCAR:
        Labels are hidden randomly.

    MAR:
        Missing probability depends on another column.
    """

    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found.")

    if not 0 < frac < 1:
        raise ValueError("frac must be between 0 and 1.")

    mechanism = mechanism.lower()

    rng = np.random.default_rng(seed)
    out = df.copy()

    if mechanism == "mcar":

        hide = rng.random(len(out)) < frac

    elif mechanism == "mar":

        if depends_on is None or depends_on not in out.columns:
            raise ValueError(
                "MAR requires a valid depends_on column."
            )

        levels = out[depends_on].dropna().unique()

        if len(levels) < 2:
            raise ValueError(
                "MAR requires at least two distinct values."
            )

        high_levels = set(
            rng.choice(
                levels,
                size=max(len(levels) // 2, 1),
                replace=False,
            )
        )

        probabilities = np.where(
            out[depends_on].isin(high_levels),
            1.5 * frac,
            0.5 * frac,
        )

        probabilities = np.clip(probabilities, 0, 1)

        hide = rng.random(len(out)) < probabilities

    else:
        raise ValueError("mechanism must be 'mcar' or 'mar'.")

    out[f"{col}_missing"] = hide

    out[f"{col}_obs"] = out[col].where(~hide, other=np.nan)

    return out


# ---------------------------------------------------------
# 7. CATEGORY SHARE TABLE
# ---------------------------------------------------------

def category_share_table(
    df: pd.DataFrame,
    window_col: str = "window",
    category_col: str = "category",
) -> pd.DataFrame:
    """
    Calculate category proportions inside each time window.
    """

    counts = (
        df.groupby([window_col, category_col])
        .size()
        .unstack(fill_value=0)
    )

    totals = counts.sum(axis=1).replace(0, np.nan)

    return counts.div(totals, axis=0)


# ---------------------------------------------------------
# 8. VISUALIZE INJECTED TRENDS
# ---------------------------------------------------------

def plot_category_share(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    category: str,
    truth: list[dict] | None = None,
    window_col: str = "window",
    category_col: str = "category",
    save_to: str | Path | None = None,
):
    """
    Compare category share before and after trend injection.
    """

    import matplotlib.pyplot as plt

    before = category_share_table(
        df_before, window_col, category_col
    )

    after = category_share_table(
        df_after, window_col, category_col
    )

    if category not in before.columns:
        raise ValueError(f"Category '{category}' not found.")

    if category not in after.columns:
        raise ValueError(f"Category '{category}' not found after injection.")

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        before.index,
        before[category],
        label="Before injection",
        alpha=0.7,
    )

    ax.plot(
        after.index,
        after[category],
        label="After injection",
        linewidth=2,
    )

    if truth:
        for record in truth:

            if (
                record["category"] == category
                and record["window"] in after.index
            ):
                position = list(after.index).index(
                    record["window"]
                )

                ax.axvspan(
                    position - 0.5,
                    position + 0.5,
                    color="orange",
                    alpha=0.2,
                )

    step = max(len(after) // 12, 1)

    ax.set_xticks(range(0, len(after), step))
    ax.set_xticklabels(
        after.index[::step],
        rotation=45,
        ha="right",
    )

    ax.set_ylabel("Category share")
    ax.set_xlabel("Time window")
    ax.set_title(
        f"{category}: category share over time"
    )

    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()

    if save_to:
        save_path = Path(save_to)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)

    return fig


# ---------------------------------------------------------
# 9. SELF-TEST
# ---------------------------------------------------------

def _synthetic(
    n_months: int = 48,
    per_month: int = 800,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Generate synthetic news data for testing.
    """

    rng = np.random.default_rng(seed)

    categories = [
        "POLITICS",
        "TRAVEL",
        "FOOD",
        "SPORTS",
        "STYLE",
        "WELLNESS",
    ]

    probabilities = np.array([
        0.30, 0.10, 0.12, 0.12, 0.16, 0.20
    ])

    rows = []

    start = pd.Timestamp("2013-01-01")

    for month in range(n_months):

        month_start = start + pd.DateOffset(months=month)

        days = rng.integers(0, 28, per_month)

        rows.append(pd.DataFrame({
            "date": month_start + pd.to_timedelta(days, unit="D"),
            "category": rng.choice(
                categories,
                per_month,
                p=probabilities,
            ),
        }))

    df = pd.concat(rows, ignore_index=True)

    df = df.sort_values("date").reset_index(drop=True)

    df["text"] = "headline " + df.index.astype(str)
    df["id"] = np.arange(len(df))

    return df


def main():

    print("\nSTREAM MODULE SELF-TEST\n")

    if DATA_PATH.exists():

        print(f"Loading dataset: {DATA_PATH}")

        df_original = pd.read_parquet(DATA_PATH)

    else:

        print("Dataset not found. Using synthetic test data.")

        df_original = _synthetic()

    # Create chronological windows.
    df = add_window(df_original, freq="M")

    # Check batch sizes.
    print("\nBATCH REPORT")

    report = window_report(df)

    print(report.head())

    # Inject artificial trends.
    categories = df["category"].value_counts().index.tolist()
    months = sorted(df["window"].unique())

    if len(months) < 37 or len(categories) < 3:
        raise ValueError(
            "Self-test needs at least 37 windows and 3 categories."
        )

    trends = [
        (categories[0], months[20:23], 2.0),
        (categories[2], months[35:37], 1.8),
    ]

    df_injected, truth = inject_trends(
        df,
        trends,
        seed=0,
    )

    print("\nINJECTED TRENDS")

    for record in truth:
        print(record)

    save_truth(truth)

    # Verify that injection did not create duplicate rows.
    assert set(df_injected["id"]).issubset(set(df["id"]))
    assert not df_injected["id"].duplicated().any()

    # Verify that untouched windows remain unchanged.
    changed_windows = {
        record["window"] for record in truth
    }

    untouched_windows = set(months) - changed_windows

    original_sizes = (
        df[df["window"].isin(untouched_windows)]
        .groupby("window")
        .size()
    )

    injected_sizes = (
        df_injected[df_injected["window"].isin(untouched_windows)]
        .groupby("window")
        .size()
    )

    assert original_sizes.equals(injected_sizes)

    # Test missing labels.
    df_missing = inject_missing(
        df_injected,
        col="category",
        frac=0.15,
        seed=1,
    )

    missing_rate = df_missing["category_missing"].mean()

    assert 0.13 < missing_rate < 0.17

    print(f"\nMissing label rate: {missing_rate:.3f}")

    # Test chronological batch generation.
    batches = list(
        stream(df_missing, min_batch=200)
    )

    assert len(batches) > 0

    labels = [label for label, _ in batches]

    assert labels == sorted(labels)

    print(f"\nBatches processed: {len(batches)}")

    # Save trend verification plot.
    plot_path = RESULTS_DIR / "stream" / "trend_check.png"

    plot_category_share(
        df,
        df_injected,
        categories[0],
        truth=truth,
        save_to=plot_path,
    )

    print(f"\nTrend verification plot: {plot_path}")
    print("\nALL STREAM TESTS PASSED")


if __name__ == "__main__":
    main()
