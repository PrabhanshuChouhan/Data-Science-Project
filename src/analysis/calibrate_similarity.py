import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from src.data_loader import load_embeddings

BATCH_DIR = Path("src/results/streaming/batches")

embeddings = load_embeddings()

# Reconstruct the original January 2012 topics
first_batch = pd.read_parquet(
    BATCH_DIR / "2012-01_articles.parquet"
)

first_ids = first_batch["id"].to_numpy()
first_embeddings = normalize(embeddings[first_ids])

model = KMeans(
    n_clusters=8,
    random_state=42,
    n_init=18,
)

model.fit(first_embeddings)

initial_centroids = normalize(model.cluster_centers_)

all_similarities = []
monthly_results = []

# Compare every month against the original eight centroids
for file in sorted(BATCH_DIR.glob("*_articles.parquet")):
    batch = pd.read_parquet(file)

    ids = batch["id"].to_numpy()
    batch_embeddings = normalize(embeddings[ids])

    similarities = batch_embeddings @ initial_centroids.T
    best = similarities.max(axis=1)

    all_similarities.extend(best.tolist())

    monthly_results.append({
        "month": file.stem.replace("_articles", ""),
        "articles": len(best),
        "median_similarity": round(float(np.median(best)), 4),
        "above_050_pct": round(float((best >= 0.50).mean() * 100), 2),
        "above_060_pct": round(float((best >= 0.60).mean() * 100), 2),
        "above_075_pct": round(float((best >= 0.75).mean() * 100), 2),
    })

all_similarities = np.asarray(all_similarities)

print("OVERALL SIMILARITY DISTRIBUTION")

for percentile in [10, 25, 50, 75, 90, 95]:
    print(
        f"{percentile}th percentile:",
        round(float(np.percentile(all_similarities, percentile)), 4)
    )

print("\nMONTHLY SIMILARITY RESULTS")

result = pd.DataFrame(monthly_results)
print(result.to_string(index=False))