import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from src.data_loader import load_embeddings

batch = pd.read_parquet(
    "src/results/streaming/batches/2012-01_articles.parquet"
)

embeddings = load_embeddings()
ids = batch["id"].to_numpy()

X = normalize(embeddings[ids])

model = KMeans(
    n_clusters=8,
    random_state=42,
    n_init=18
)

labels = model.fit_predict(X)
centroids = normalize(model.cluster_centers_)

# Similarity to each article's own assigned centroid
similarities = np.sum(
    X * centroids[labels],
    axis=1
)

print("INITIAL CLUSTER SIMILARITY")
print("Articles:", len(similarities))

for p in [0, 10, 20, 25, 50, 75, 90, 100]:
    print(
        f"{p}th percentile:",
        round(float(np.percentile(similarities, p)), 4)
    )

print("\nCoverage at different thresholds")

for threshold in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    coverage = (similarities >= threshold).mean() * 100
    print(f"{threshold}: {coverage:.2f}%")