import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from sklearn.cluster import kmeans_plusplus, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import normalized_mutual_info_score
from sklearn.feature_extraction.text import TfidfVectorizer
from src.stream import inject_missing_labels

class DiagGMM:
    def __init__(self, K, iters=100, tol=1e-5, reg=1e-4, seed=0):
        self.K, self.iters, self.tol, self.reg, self.seed = K, iters, tol, reg, seed

    def _log_joint(self, X):
        # (n, K): log pi_k + log N(x | mu_k, var_k)
        var = self.var
        ll = -0.5 * (
            np.log(2 * np.pi * var).sum(1)[None, :]
            + (((X[:, None, :] - self.mu[None, :, :]) ** 2) / var[None, :, :]).sum(2)
        )
        return ll + np.log(self.pi)[None, :]

    def fit(self, X, init=None):
        n, d = X.shape
        if init is None:
            self.mu, _ = kmeans_plusplus(X, self.K, random_state=self.seed)
            self.var = np.tile(X.var(0) + self.reg, (self.K, 1))
            self.pi = np.full(self.K, 1 / self.K)
        else:
            self.pi, self.mu, self.var = [a.copy() for a in init]  # warm start

        self.ll_hist, prev = [], -np.inf
        for _ in range(self.iters):
            lj = self._log_joint(X)  # E-step
            lse = logsumexp(lj, axis=1)
            R = np.exp(lj - lse[:, None])
            ll = lse.sum()
            self.ll_hist.append(ll)

            # M-step
            Nk = R.sum(0) + 1e-10
            self.pi = Nk / n
            self.mu = (R.T @ X) / Nk[:, None]
            self.var = np.maximum((R.T @ (X ** 2)) / Nk[:, None] - self.mu ** 2, self.reg)

            if abs(ll - prev) < self.tol * abs(ll):
                break
            prev = ll
        return self

    def resp(self, X):
        lj = self._log_joint(X)
        return np.exp(lj - logsumexp(lj, axis=1)[:, None])

    def predict(self, X):
        return self.resp(X).argmax(1)

    def bic(self, X):
        n, d = X.shape
        p = (self.K - 1) + 2 * self.K * d
        return -2 * logsumexp(self._log_joint(X), axis=1).sum() + p * np.log(n)


if __name__ == "__main__":
    df = pd.read_parquet("data/processed/news.parquet")
    Z = np.load("data/processed/emb_pca50.npy")

    # Warm-up data: first 12 months
    warmup_months = sorted(df.month.unique())[:12]
    warm_mask = df.month.isin(warmup_months).values
    Z_warm = Z[warm_mask]
    y_warm = df.loc[warm_mask, "category"].values

    # 1. Sanity check: log-likelihood monotonic & compare with sklearn GMM
    gmm_test = DiagGMM(K=8, seed=0).fit(Z_warm)
    diffs = np.diff(gmm_test.ll_hist)
    print("Log-likelihood strictly non-decreasing:", np.all(diffs >= -1e-6))

    sk_gmm = GaussianMixture(n_components=8, covariance_type="diag", random_state=0).fit(Z_warm)
    nmi_diag = normalized_mutual_info_score(y_warm, gmm_test.predict(Z_warm))
    nmi_sk = normalized_mutual_info_score(y_warm, sk_gmm.predict(Z_warm))
    nmi_km = normalized_mutual_info_score(y_warm, KMeans(8, n_init=10, random_state=0).fit_predict(Z_warm))

    # 2. Choose K using BIC on warm-up data
    Ks = [4, 6, 8, 10, 12, 16]
    bics = [DiagGMM(K=k, seed=0).fit(Z_warm).bic(Z_warm) for k in Ks]
    pd.DataFrame({"K": Ks, "BIC": bics}).to_csv("results/phase5_bic.csv", index=False)

    plt.figure(figsize=(7, 4))
    plt.plot(Ks, bics, marker="o")
    plt.title("BIC Score vs Number of Topics K (Warm-up Period)")
    plt.xlabel("Number of Topics (K)")
    plt.ylabel("BIC (lower is better)")
    plt.tight_layout()
    plt.savefig("results/phase5_bic.png")
    plt.close()

    # 3. Compare NMI against sklearn GMM and k-means
    nmi_df = pd.DataFrame({
        "method": ["DiagGMM (from scratch)", "sklearn GaussianMixture", "K-Means"],
        "NMI": [nmi_diag, nmi_sk, nmi_km]
    })
    nmi_df.to_csv("results/phase5_em_vs_kmeans.csv", index=False)
    print("\nNMI Comparison:\n", nmi_df)

    # 4. Fix topic identities on warm-up months & name topics via top TF-IDF words
    K_best = 8
    gmm = DiagGMM(K=K_best, seed=0).fit(Z_warm)
    df["topic"] = gmm.predict(Z[df.id.values])

    vec = TfidfVectorizer(max_features=5000, stop_words="english")
    X_tf = vec.fit_transform(df.text)
    terms = np.array(vec.get_feature_names_out())
    topic_names = {}
    print("\nDiscovered Topics (Top TF-IDF keywords):")
    for k in range(K_best):
        top_idx = np.asarray(X_tf[df.topic.values == k].mean(axis=0)).ravel().argsort()[-5:][::-1]
        topic_names[k] = ", ".join(terms[top_idx])
        print(f"  Topic {k}: {topic_names[k]}")

    # 5. Warm-start vs cold-start convergence speed across later batches
    later_months = sorted(df.month.unique())[12:24]
    prev_params = (gmm.pi, gmm.mu, gmm.var)
    conv_rows = []
    for m in later_months:
        Zm = Z[df.month == m]
        warm_m = DiagGMM(K=K_best, seed=0).fit(Zm, init=prev_params)
        cold_m = DiagGMM(K=K_best, seed=0).fit(Zm, init=None)
        prev_params = (warm_m.pi, warm_m.mu, warm_m.var)
        conv_rows.append({"month": m, "warm_iters": len(warm_m.ll_hist), "cold_iters": len(cold_m.ll_hist)})
    conv_df = pd.DataFrame(conv_rows)
    conv_df.to_csv("results/phase5_warm_vs_cold.csv", index=False)
    print(f"\nAvg iterations — Warm start: {conv_df.warm_iters.mean():.1f} | Cold start: {conv_df.cold_iters.mean():.1f}")

    # 6. Imputation of hidden labels using EM posterior responsibilities
    df_miss = inject_missing_labels(df, rate=0.15, seed=1)
    R = gmm.resp(Z)
    obs = df_miss.category_obs.notna().values
    cats = sorted(df_miss.category.unique())
    Pl = np.array([[R[obs & (df_miss.category_obs == c).values, k].sum() for c in cats] for k in range(gmm.K)])
    Pl /= Pl.sum(1, keepdims=True)
    pred = np.array(cats)[(R[~obs] @ Pl).argmax(1)]
    acc = (pred == df_miss.category.values[~obs]).mean()
    print(f"EM posterior imputation accuracy (15% missing): {acc:.4f}")

    # 7. Save topic shares per month (input for Phase 7 trend detection)
    shares = df.groupby(["month", "topic"]).size().rename("count").reset_index()
    shares["share"] = shares["count"] / shares.groupby("month")["count"].transform("sum")
    shares.to_csv("results/topic_shares.csv", index=False)
    print("Phase 5 complete! Saved results/topic_shares.csv and all evaluation tables.")
