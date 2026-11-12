import time
import numpy as np
import pandas as pd
import faiss
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE

base = pd.read_parquet("data/processed/news.parquet")
E_all = np.load("data/processed/emb.npy")

# (a) PCA via SVD (from scratch) on a 30k sample
mu = E_all.mean(0)
sample = E_all[np.random.default_rng(0).choice(len(E_all), 30000, replace=False)] - mu
U, S, Vt = np.linalg.svd(sample, full_matrices=False)
explained = np.cumsum(S**2) / np.sum(S**2)
rank_err = 1 - explained  # relative low-rank approximation error

# Verify against sklearn
sk_pca = PCA(50).fit(sample)
print("Max diff vs sklearn explained variance (top 50):",
      np.max(np.abs((S[:50]**2 / np.sum(S**2)) - sk_pca.explained_variance_ratio_)))

def project(X, k=50):
    return (X - mu) @ Vt[:k].T

Z = project(E_all, 50)  # (N, 50)
np.save("data/processed/emb_pca50.npy", Z.astype("float32"))
np.savez("data/processed/pca_params.npz", mu=mu.astype("float32"), Vt=Vt[:50].astype("float32"))

# Plot explained variance and low-rank approximation error
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(np.arange(1, len(explained) + 1), explained)
ax[0].set(title="PCA Explained Variance", xlabel="Rank k", ylabel="Cumulative Variance")
ax[1].plot(np.arange(1, len(rank_err) + 1), rank_err)
ax[1].set(title="Low-Rank Approximation Error (1 - Explained)", xlabel="Rank k", ylabel="Relative Error")
plt.tight_layout()
plt.savefig("results/phase4_pca_variance.png")
plt.close()

# (b) Retrieval versus k
idx = np.random.default_rng(0).choice(len(base), 2000, replace=False)
y = base.category.values[idx]

def knn_acc(X, metric="cosine", k=10):
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(X)
    nbrs = nn.kneighbors(X, return_distance=False)[:, 1:]
    return np.mean([Counter(y[n]).most_common(1)[0][0] == y[i] for i, n in enumerate(nbrs)])

ks = [5, 10, 20, 50, 100, 384]
accs = [knn_acc(project(E_all[idx], k) if k < 384 else E_all[idx]) for k in ks]
pd.DataFrame({"k": ks, "knn_acc": accs}).to_csv("results/phase4_retrieval_vs_k.csv", index=False)

plt.figure(figsize=(7, 4))
plt.plot(ks, accs, marker="o")
plt.title("Retrieval Accuracy (kNN@10) vs PCA Rank k")
plt.xlabel("Dimensions (k)")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig("results/phase4_retrieval_vs_k.png")
plt.close()

# (c) LSH from scratch (random hyperplanes) with FAISS as reference
class LSH:
    def __init__(self, d, bits=12, tables=8, seed=0):
        rng = np.random.default_rng(seed)
        self.P = [rng.standard_normal((bits, d)) for _ in range(tables)]
        self.T = [dict() for _ in range(tables)]

    def _keys(self, X, P):
        return [r.tobytes() for r in (X @ P.T > 0)]

    def add(self, X):
        for P, T in zip(self.P, self.T):
            for i, k in enumerate(self._keys(X, P)):
                T.setdefault(k, []).append(i)

    def candidates(self, x):
        c = set()
        for P, T in zip(self.P, self.T):
            c.update(T.get(self._keys(x[None], P)[0], []))
        return list(c)

Xn = E_all / np.linalg.norm(E_all, axis=1, keepdims=True)
Xf = Xn.astype("float32")
index = faiss.IndexFlatIP(Xf.shape[1])
index.add(Xf)

t0 = time.perf_counter()
D, I = index.search(Xf[:200], 11)
faiss_ms = (time.perf_counter() - t0) * 1000

lsh_rows = []
for bits, tables in [(8, 4), (12, 8), (12, 16), (16, 16)]:
    lsh = LSH(Xn.shape[1], bits=bits, tables=tables)
    lsh.add(Xn)
    t0 = time.perf_counter()
    recalls, cand_counts = [], []
    for q in range(200):
        cands = np.array(lsh.candidates(Xn[q]))
        cand_counts.append(len(cands))
        if len(cands) > 0:
            sims = Xn[cands] @ Xn[q]
            top10 = cands[np.argsort(-sims)[:11]]
            recalls.append(len(set(top10) & set(I[q, 1:])) / 10.0)
        else:
            recalls.append(0.0)
    lsh_ms = (time.perf_counter() - t0) * 1000
    lsh_rows.append({
        "bits": bits,
        "tables": tables,
        "recall@10": np.mean(recalls),
        "avg_candidates": np.mean(cand_counts),
        "candidate_ratio": np.mean(cand_counts) / len(Xn),
        "query_ms_200": lsh_ms,
        "faiss_ms_200": faiss_ms,
    })

lsh_df = pd.DataFrame(lsh_rows)
lsh_df.to_csv("results/phase4_lsh_comparison.csv", index=False)
print("\nLSH vs FAISS:\n", lsh_df)

# (d) Near-duplicate removal per batch
def dedup(Xb, thr=0.95):
    Xb = Xb / np.linalg.norm(Xb, axis=1, keepdims=True)
    S = np.triu(Xb @ Xb.T, k=1)
    return ~(S > thr).any(axis=0)  # boolean mask: keep items with no earlier near-duplicate

kept_total, total_items = 0, 0
for _, g in base.groupby("month", sort=True):
    mask = dedup(Z[g.id.values], thr=0.95)
    kept_total += mask.sum()
    total_items += len(g)
pct_dropped = 100 * (1 - kept_total / total_items)
print(f"\nNear-duplicates dropped across stream: {pct_dropped:.2f}%")

# (e) t-SNE on 3,000 items from one batch
largest_month = base.month.value_counts().idxmax()
b_idx = base[base.month == largest_month].id.values[:3000]
tsne_xy = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(Z[b_idx])
tsne_df = pd.DataFrame({
    "x": tsne_xy[:, 0],
    "y": tsne_xy[:, 1],
    "category": base.loc[b_idx, "category"].values
})
tsne_df.to_csv("results/tsne_sample.csv", index=False)

plt.figure(figsize=(9, 6))
for cat, grp in tsne_df.groupby("category"):
    plt.scatter(grp.x, grp.y, s=8, alpha=0.7, label=cat)
plt.legend(markerscale=2, bbox_to_anchor=(1.02, 1), loc="upper left")
plt.title(f"t-SNE Map of Batch {largest_month} (PCA-50)")
plt.tight_layout()
plt.savefig("results/phase4_tsne.png")
plt.close()
print("Phase 4 complete! Saved all plots and CSVs to results/.")
