import numpy as np
import joblib

from pathlib import Path
from sklearn.preprocessing import normalize

CHECKPOINT = Path(
    "src/results/streaming/checkpoints/engine_state.joblib"
)

state = joblib.load(CHECKPOINT)

centroids = normalize(
    np.asarray(state["topic_centroids"], dtype=float)
)

print("Number of stored centroids:", len(centroids))
print("Centroid dimensions:", centroids.shape[1])

# Load the first monthly batch embeddings
from src.data_loader import load_embeddings

embeddings = load_embeddings()

# The first batch contains January 2012 article IDs
import pandas as pd

first_batch = pd.read_parquet(
    "src/results/streaming/batches/2012-01_articles.parquet"
)

article_ids = first_batch["id"].to_numpy()

first_embeddings = normalize(embeddings[article_ids])

similarities = first_embeddings @ centroids.T

best_similarities = similarities.max(axis=1)

print("\nSimilarity distribution:")
for percentile in [0, 10, 25, 50, 75, 90, 95, 99, 100]:
    print(
        f"{percentile}th percentile:",
        round(np.percentile(best_similarities, percentile), 4)
    )

print("\nAssignment percentages at different thresholds:")
for threshold in [0.30, 0.40, 0.50, 0.60, 0.70, 0.75]:
    percentage = (best_similarities >= threshold).mean() * 100
    print(f"{threshold}: {percentage:.2f}%")