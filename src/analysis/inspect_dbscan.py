import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import normalize

from src.data_loader import load_embeddings

BATCH_DIR = "src/results/streaming/batches"

embeddings = load_embeddings()

# Reconstruct initial topics
first = pd.read_parquet(
    f"{BATCH_DIR}/2012-01_articles.parquet"
)

X_first = normalize(
    embeddings[first["id"].to_numpy()]
)

kmeans = KMeans(
    n_clusters=8,
    random_state=42,
    n_init=18
)

kmeans.fit(X_first)

centroids = normalize(kmeans.cluster_centers_)

# Inspect May 2014
for month in ["2014-05", "2015-01", "2017-01"]:

    batch = pd.read_parquet(
        f"{BATCH_DIR}/{month}_articles.parquet"
    )

    X = normalize(
        embeddings[batch["id"].to_numpy()]
    )

    similarities = X @ centroids.T
    best = similarities.max(axis=1)

    unfamiliar = X[best < 0.30]

    print(f"\n{'=' * 50}")
    print("MONTH:", month)
    print("Total articles:", len(X))
    print("Unfamiliar articles:", len(unfamiliar))

    for eps in [0.40, 0.45, 0.50]:

        model = DBSCAN(
            eps=eps,
            min_samples=3,
            metric="cosine"
        )

        labels = model.fit_predict(unfamiliar)
        if month == "2017-01" and eps == 0.45:

            largest_cluster_id = max(
                set(labels) - {-1},
                key=lambda cluster_id: np.sum(labels == cluster_id)
            )

            cluster_positions = np.where(
                labels == largest_cluster_id
            )[0]

            unfamiliar_ids = batch["id"].to_numpy()[
                np.where(best < 0.30)[0]
            ]

            cluster_article_ids = unfamiliar_ids[cluster_positions]

            articles = pd.read_parquet(
                "data/processed/news.parquet"
            )

            selected = articles[
                articles["id"].isin(cluster_article_ids)
            ]

            print("\nLARGEST CLUSTER ARTICLE SAMPLES:")

            for _, row in selected.head(20).iterrows():
                print("\nCategory:", row["category"])
                print("Text:", str(row["text"])[:400])
       
        
     