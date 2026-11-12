import torch
import numpy as np
import pandas as pd
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import NearestNeighbors

# 1. Embeddings on NVIDIA GPU (cached once)
base = pd.read_parquet("data/processed/news.parquet")
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Encoding on device:", device)

model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
emb = model.encode(base.text.tolist(), batch_size=256, show_progress_bar=True)  # (N, 384)
np.save("data/processed/emb.npy", emb.astype("float32"))

# 2. TF-IDF baseline
tfidf = TfidfVectorizer(max_features=20000, stop_words="english", norm=None)
X_tfidf = tfidf.fit_transform(base.text)

# 3. Compare distance measures on a 2,000-item sample
idx = np.random.default_rng(0).choice(len(base), 2000, replace=False)
y = base.category.values[idx]

def knn_acc(X, metric, k=10):
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(X)
    nbrs = nn.kneighbors(X, return_distance=False)[:, 1:]
    return np.mean([Counter(y[n]).most_common(1)[0][0] == y[i] for i, n in enumerate(nbrs)])

E = emb[idx]
results = {
    "embed-cosine": knn_acc(E, "cosine"),
    "embed-euclid": knn_acc(E, "euclidean"),
    "tfidf-cosine": knn_acc(X_tfidf[idx], "cosine"),
    "bow-jaccard": knn_acc(
        CountVectorizer(binary=True, max_features=5000)
        .fit_transform(base.text.values[idx])
        .toarray()
        .astype(bool),
        "jaccard",
    ),
}
print(results)

pd.DataFrame.from_dict(results, orient="index", columns=["knn_acc"]).to_csv(
    "results/phase3_representation_results.csv"
)
