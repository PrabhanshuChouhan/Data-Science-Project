import numpy as np

def stream(df):
    for month, g in df.groupby("month", sort=True):
        yield month, g

def inject_trends(df, trends, seed=0):
    """trends: list of (category, [months], factor). Returns (new_df, ground_truth)."""
    rng = np.random.default_rng(seed)
    keep = np.ones(len(df), dtype=bool)
    truth = []
    for cat, months, f in trends:
        for m in months:
            in_m = (df.month == m).values
            is_c = (df.category == cat).values
            n_t, n_all = (in_m & is_c).sum(), in_m.sum()
            target_share = min(f * n_t / n_all, 0.9)
            n_others_keep = int(n_t / target_share - n_t)
            others = np.where(in_m & ~is_c)[0]
            drop = rng.choice(others, size=max(len(others) - n_others_keep, 0), replace=False)
            keep[drop] = False
            truth.append((cat, m))
    return df[keep].reset_index(drop=True), truth

def inject_missing_labels(df, rate=0.15, seed=1):
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["category_obs"] = df["category"]
    df.loc[rng.random(len(df)) < rate, "category_obs"] = np.nan
    return df
