import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from src.data_loader import load_embeddings

BATCH_DIR = Path("src/results/streaming/batches")

embeddings = load_embeddings()

first_batch = pd.read_parquet(
    BATCH_DIR / "2012-01_articles.parquet"
)

first_ids = first_batch["id"].to_numpy()
X = normalize(embeddings[first_ids])

model = KMeans(
    n_clusters=8,
    random_state=42,
    n_init=18
)

labels = model.fit_predict(X)

initial_centroids = normalize(model.cluster_centers_)
initial_counts = np.bincount(labels, minlength=8)

thresholds = [0.30, 0.35, 0.40, 0.45]

results = {}

for threshold in thresholds:

    centroids = initial_centroids.copy()
    counts = initial_counts.astype(float).copy()

    monthly_results = []

    for file in sorted(BATCH_DIR.glob("*_articles.parquet")):

        if file.stem == "2012-01_articles":
            continue

        batch = pd.read_parquet(file)
        ids = batch["id"].to_numpy()

        X = normalize(embeddings[ids])

        similarities = X @ centroids.T

        best_ids = np.argmax(similarities, axis=1)
        best_scores = np.max(similarities, axis=1)

        assigned = best_scores >= threshold

        # Update centroids using assigned articles
        for topic_id in np.unique(best_ids[assigned]):

            mask = assigned & (best_ids == topic_id)
            new_embeddings = X[mask]

            new_count = len(new_embeddings)
            new_mean = new_embeddings.mean(axis=0)

            updated = (
                centroids[topic_id] * counts[topic_id]
                + new_mean * new_count
            ) / (counts[topic_id] + new_count)

            centroids[topic_id] = normalize(
                updated.reshape(1, -1)
            )[0]

            counts[topic_id] += new_count

        monthly_results.append({
            "month": file.stem.replace("_articles", ""),
            "articles": len(X),
            "assigned_pct": round(assigned.mean() * 100, 2),
        })

    results[threshold] = pd.DataFrame(monthly_results)

print("\nONLINE MATCHING RESULTS")

for threshold, result in results.items():

    print(f"\nThreshold: {threshold}")
    print(
        "Average assigned:",
        round(result["assigned_pct"].mean(), 2),
        "%"
    )
    print(
        "Last month assigned:",
        result.iloc[-1]["assigned_pct"],
        "%"
    )

print("\nMONTHLY COMPARISON")

comparison = pd.DataFrame({
    str(threshold): result.set_index("month")["assigned_pct"]
    for threshold, result in results.items()
})

print(comparison.to_string())