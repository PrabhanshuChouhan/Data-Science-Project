from pathlib import Path
import pandas as pd
import numpy as np

from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import normalize

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "processed"
NEWS_PATH = DATA_DIR / "news.parquet"
EMBEDDINGS_PATH = DATA_DIR / "emb.npy"

OUTPUT_DIR = BASE_DIR / "results" / "streaming" / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load news articles and embeddings."""
    
    news = pd.read_parquet(NEWS_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)
    
    if len(news) != embeddings.shape[0]:
        raise ValueError(f"Number of embeddings ({embeddings.shape[0]}) does not match number of articles ({len(news)}).")
    ids=news["id"].to_numpy(dtype=int)
    embeddings=embeddings[ids]
    
    news = news.copy()
    news["date"] = pd.to_datetime(
        news["date"],
        errors="coerce",
    )

    if news["date"].isna().any():
        raise ValueError("Invalid dates found in news data.")

    news["month"] = (
        news["date"].dt.to_period("M").astype(str)
    )

    return news, embeddings

def calculate_monthly_centroids(news, embeddings):
    """Calculate monthly centroids of embeddings."""
    
    monthly_centroids = {}

    for month, indices in news.groupby("month").groups.items():

        positions = news.index.get_indexer(indices)

        monthly_embeddings = embeddings[positions]

        centroid = monthly_embeddings.mean(axis=0)

        monthly_centroids[month] = normalize(
            centroid.reshape(1, -1)
        )[0]

    return monthly_centroids


def calculate_centroid_drift(monthly_centroids):
    """Measure semantic movement between consecutive months."""

    months = sorted(monthly_centroids.keys())

    records = []

    for index, month in enumerate(months):

        current = monthly_centroids[month]

        if index == 0:
            records.append({
                "month": month,
                "previous_month": None,
                "cosine_similarity": None,
                "semantic_drift": None,
            })
            continue

        previous_month = months[index - 1]
        previous = monthly_centroids[previous_month]

        similarity = float(np.dot(previous, current))

        similarity = np.clip(similarity, -1.0, 1.0)

        drift = 1.0 - similarity

        records.append({
            "month": month,
            "previous_month": previous_month,
            "cosine_similarity": similarity,
            "semantic_drift": drift,
        })

    return pd.DataFrame(records)


def calculate_distribution_drift(news, embeddings, bins=20):
    """
    Compare monthly embedding distributions against
    a historical baseline using Jensen-Shannon distance.

    The embedding space is projected onto a fixed PCA axis
    before comparing monthly distributions.
    """

    from sklearn.decomposition import PCA

    normalized_embeddings = normalize(embeddings)

    # Fit one fixed projection on the full historical dataset.
    # This is descriptive retrospective analysis, not a
    # leakage-free online forecasting evaluation.
    pca = PCA(n_components=1, random_state=42)

    projected = pca.fit_transform(
        normalized_embeddings
    ).ravel()

    news = news.copy()
    news["projection"] = projected

    minimum = float(projected.min())
    maximum = float(projected.max())

    if minimum == maximum:
        return pd.DataFrame({
            "month": sorted(news["month"].unique()),
            "distribution_drift": 0.0,
        })

    edges = np.linspace(minimum, maximum, bins + 1)

    monthly_distributions = {}

    for month, group in news.groupby("month"):

        counts, _ = np.histogram(
            group["projection"],
            bins=edges,
        )

        distribution = counts.astype(float) + 1e-8
        distribution /= distribution.sum()

        monthly_distributions[month] = distribution

    baseline_months = sorted(
        monthly_distributions.keys()
    )[:3]

    baseline = np.mean(
        [
            monthly_distributions[month]
            for month in baseline_months
        ],
        axis=0,
    )

    records = []

    for month in sorted(monthly_distributions):

        distance = float(
            jensenshannon(
                baseline,
                monthly_distributions[month],
            )
        )

        records.append({
            "month": month,
            "distribution_drift": distance,
        })

    return pd.DataFrame(records)


def run_semantic_drift():

    print("Loading news and embeddings...")

    news, embeddings = load_data()

    print(f"Articles loaded: {len(news)}")

    print("Calculating monthly semantic centroids...")

    centroids = calculate_monthly_centroids(
        news,
        embeddings,
    )

    centroid_drift = calculate_centroid_drift(
        centroids
    )

    print("Calculating distribution drift...")

    distribution_drift = calculate_distribution_drift(
        news,
        embeddings,
    )

    results = centroid_drift.merge(
        distribution_drift,
        on="month",
        how="left",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        OUTPUT_DIR / "semantic_drift.csv",
        index=False,
    )

    print("\nSemantic drift analysis completed.")
    print(f"Months analyzed: {len(results)}")

    print("\nLatest semantic drift:")

    print(
        results.tail(10).to_string(index=False)
    )

    print("\nResults saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    run_semantic_drift()
