import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import betaln, gammaln
from src.stream import inject_trends

def benjamini_hochberg(pvals):
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty_like(q)
    out[order] = q
    return out

def bayes_factor_spike(k_curr, n_curr, k_prev, n_prev):
    """Bayes Factor BF_10 comparing an independent surge rate (H1) vs baseline rate (H0)."""
    if n_curr == 0 or (k_curr / n_curr) <= (k_prev / max(n_prev, 1)):
        return 0.0
    log_comb = gammaln(n_curr + 1) - gammaln(k_curr + 1) - gammaln(n_curr - k_curr + 1)
    log_p_h0 = (
        log_comb
        + betaln(1 + k_prev + k_curr, 1 + (n_prev - k_prev) + (n_curr - k_curr))
        - betaln(1 + k_prev, 1 + (n_prev - k_prev))
    )
    log_p_h1 = -np.log(n_curr + 1)  # Beta(1,1) marginal likelihood
    return float(np.exp(np.clip(log_p_h1 - log_p_h0, -50, 50)))

def detect_trends(df, group_col="category", window=3, n_perm=1000, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    months = sorted(df["month"].unique())
    groups = sorted(df[group_col].dropna().unique())

    month_counts = df.groupby("month").size().to_dict()
    cell_counts = df.groupby(["month", group_col]).size().to_dict()

    rows = []
    for t in range(window, len(months)):
        m_curr = months[t]
        prev_months = months[t - window : t]
        n_curr = month_counts.get(m_curr, 0)
        n_prev = sum(month_counts.get(pm, 0) for pm in prev_months)
        if n_curr == 0 or n_prev == 0:
            continue

        for g in groups:
            k_curr = cell_counts.get((m_curr, g), 0)
            k_prev = sum(cell_counts.get((pm, g), 0) for pm in prev_months)

            p_curr = k_curr / n_curr
            p_prev = k_prev / n_prev
            diff = p_curr - p_prev

            if diff <= 0:
                p_val = 1.0
                ci_low, ci_high = diff, diff
            else:
                perm_k_curr = rng.hypergeometric(
                    ngood=k_curr + k_prev,
                    nbad=(n_curr + n_prev) - (k_curr + k_prev),
                    nsample=n_curr,
                    size=n_perm,
                )
                perm_diffs = (perm_k_curr / n_curr) - ((k_curr + k_prev - perm_k_curr) / n_prev)
                p_val = (1.0 + np.sum(perm_diffs >= diff)) / (n_perm + 1.0)

                boot_curr = rng.binomial(n_curr, p_curr, size=n_boot) / n_curr
                boot_prev = rng.binomial(n_prev, p_prev, size=n_boot) / n_prev
                boot_diffs = boot_curr - boot_prev
                ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])

            bf10 = bayes_factor_spike(k_curr, n_curr, k_prev, n_prev)
            rows.append({
                group_col: g,
                "month": m_curr,
                "p_prev": round(p_prev, 4),
                "p_curr": round(p_curr, 4),
                "diff": round(diff, 4),
                "ci_95_low": round(float(ci_low), 4),
                "ci_95_high": round(float(ci_high), 4),
                "p_value": p_val,
                "bf10": bf10,
            })

    res = pd.DataFrame(rows)
    res["p_bonferroni"] = np.minimum(1.0, res["p_value"] * len(res))
    res["q_bh"] = benjamini_hochberg(res["p_value"].values)
    return res

def evaluate_against_truth(res, truth_pairs, alpha=0.05):
    truth_set = {tuple(x) for x in truth_pairs}
    eval_rows = []
    methods = {
        "Uncorrected (p < 0.05)": res["p_value"] < alpha,
        "Benjamini-Hochberg (q < 0.05)": res["q_bh"] < alpha,
        "Bonferroni (p_bonf < 0.05)": res["p_bonferroni"] < alpha,
        "Bayes Factor (BF10 > 10)": res["bf10"] > 10.0,
    }
    for name, mask in methods.items():
        flagged = set(zip(res.loc[mask, "category"], res.loc[mask, "month"]))
        tp = len(flagged & truth_set)
        fp = len(flagged - truth_set)
        fn = len(truth_set - flagged)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        eval_rows.append({
            "method": name,
            "flagged": len(flagged),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        })
    return pd.DataFrame(eval_rows)

if __name__ == "__main__":
    df = pd.read_parquet("data/processed/news.parquet")

    # 1. Benchmark on Phase 2 injected trends
    trends = [
        ("POLITICS", ["2016-03", "2016-04", "2016-05"], 1.8),
        ("TRAVEL", ["2014-06", "2014-07"], 2.0),
    ]
    df_sim, truth = inject_trends(df, trends)
    res_cat = detect_trends(df_sim, group_col="category", window=3, n_perm=2000, n_boot=1000)
    res_cat.to_csv("results/phase7_category_trends.csv", index=False)

    eval_df = evaluate_against_truth(res_cat, truth)
    eval_df.to_csv("results/phase7_benchmark_metrics.csv", index=False)
    print("Benchmark vs Phase 2 Ground Truth:\n", eval_df.to_string(index=False))

    sig_cat = res_cat[res_cat["q_bh"] < 0.05].sort_values("bf10", ascending=False)
    print("\nTop Significant Category Surges (BH q < 0.05):\n", sig_cat.head(10).to_string(index=False))

    # 2. Detect spikes in Phase 5 unsupervised EM topics
    topic_shares = pd.read_csv("results/topic_shares.csv")
    topic_rows = topic_shares.loc[topic_shares.index.repeat(topic_shares["count"])][["month", "topic"]].reset_index(drop=True)
    res_topic = detect_trends(topic_rows, group_col="topic", window=3, n_perm=2000, n_boot=1000)
    res_topic.to_csv("results/phase7_topic_trends.csv", index=False)

    sig_top = res_topic[res_topic["q_bh"] < 0.05].sort_values("bf10", ascending=False)
    print("\nTop Significant EM Topic Surges (BH q < 0.05):\n", sig_top.head(10).to_string(index=False))

    # 3. Plot detected BH spikes on Politics & Travel share curves
    shares = pd.crosstab(df_sim["month"], df_sim["category"], normalize="index")
    ax = shares[["POLITICS", "TRAVEL"]].plot(figsize=(10, 4), title="Phase 7 Detected Trend Spikes (BH q < 0.05)")
    for _, r in sig_cat[sig_cat["category"].isin(["POLITICS", "TRAVEL"])].iterrows():
        x_pos = list(shares.index).index(r["month"])
        y_pos = shares.loc[r["month"], r["category"]]
        ax.scatter(x_pos, y_pos, color="red", zorder=5, s=35)
    plt.tight_layout()
    plt.savefig("results/phase7_detected_spikes.png")
    plt.close()
    print("\nPhase 7 complete! Saved trend tables, benchmark metrics, and plot to results/.")
