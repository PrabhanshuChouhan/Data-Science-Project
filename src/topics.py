
"""
Topic discovery and clustering.

Algorithms:
    1. K-Means
    2. Gaussian Mixture Model (GMM, using EM internally)

Inputs:
    data/processed/news.parquet
    data/processed/pca_embeddings.npy
    data/processed/tfidf.npz
    data/processed/tfidf_vectorizer.joblib

Outputs:
    results/topics/article_topics.parquet
    results/topics/clustering_metrics.csv
    results/topics/topic_keywords.csv
    results/topics/topic_examples.csv
    results/topics/kmeans_model.joblib
    results/topics/gmm_model.joblib
    results/topics/gmm_probabilities.npy
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import load_npz

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    normalized_mutual_info_score,
)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "results" / "topics"

DATA_PATH = PROCESSED_DIR / "news.parquet"
PCA_PATH = PROCESSED_DIR / "pca_embeddings.npy"
TFIDF_PATH = PROCESSED_DIR / "tfidf.npz"
VECTORIZER_PATH = PROCESSED_DIR / "tfidf_vectorizer.joblib"

ASSIGNMENTS_PATH = OUTPUT_DIR / "article_topics.parquet"
METRICS_PATH = OUTPUT_DIR / "clustering_metrics.csv"
KEYWORDS_PATH = OUTPUT_DIR / "topic_keywords.csv"
EXAMPLES_PATH = OUTPUT_DIR / "topic_examples.csv"

KMEANS_PATH = OUTPUT_DIR / "kmeans_model.joblib"
GMM_PATH = OUTPUT_DIR / "gmm_model.joblib"
GMM_PROBABILITIES_PATH = OUTPUT_DIR / "gmm_probabilities.npy"


# ---------------------------------------------------------
# Load and validate inputs
# ---------------------------------------------------------

def load_data():
    """Load articles and their precomputed representations."""

    required_files = [
        DATA_PATH,
        PCA_PATH,
        TFIDF_PATH,
        VECTORIZER_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required input not found: {path}"
            )

    df = pd.read_parquet(DATA_PATH)
    X = np.load(PCA_PATH, mmap_mode="r")
    X_tfidf = load_npz(TFIDF_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    required_columns = {"id", "category", "text"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing columns: {missing}"
        )

    df = df.sort_values("id").reset_index(drop=True)

    expected_ids = np.arange(len(df))

    if not np.array_equal(
        df["id"].to_numpy(),
        expected_ids,
    ):
        raise ValueError(
            "Article IDs must be contiguous from 0 to N-1."
        )

    if X.ndim != 2 or X.shape[0] != len(df):
        raise ValueError(
            "PCA matrix does not align with article data."
        )

    if X_tfidf.shape[0] != len(df):
        raise ValueError(
            "TF-IDF matrix does not align with article data."
        )

    if X_tfidf.shape[1] != len(
        vectorizer.get_feature_names_out()
    ):
        raise ValueError(
            "TF-IDF vocabulary and matrix dimensions do not match."
        )

    # Validate the PCA matrix in chunks to avoid
    # allocating another large boolean matrix.
    chunk_size = 10000

    for start in range(0, len(X), chunk_size):
        chunk = X[start:start + chunk_size]

        if not np.isfinite(chunk).all():
            raise ValueError(
                f"PCA matrix contains invalid values near row {start}."
            )

    print(f"Articles: {len(df):,}")
    print(f"PCA matrix: {X.shape}")
    print(f"TF-IDF matrix: {X_tfidf.shape}")

    return df, X, X_tfidf, vectorizer


# ---------------------------------------------------------
# Clustering algorithms
# ---------------------------------------------------------

def run_kmeans(X, n_clusters=8, random_state=42):
    """Fit K-Means using scikit-learn."""

    print("\nFitting K-Means...")

    model = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=random_state,
    )

    labels = model.fit_predict(X)

    print("K-Means completed.")
    print(f"Inertia: {model.inertia_:.4f}")

    return model, labels


def run_gmm(X, n_components=8, random_state=42):
    """Fit Gaussian Mixture Model using scikit-learn EM."""

    print("\nFitting Gaussian Mixture Model...")

    model = GaussianMixture(
        n_components=n_components,
        covariance_type="diag",
        n_init=2,
        max_iter=200,
        reg_covar=1e-5,
        random_state=random_state,
    )

    model.fit(X)

    labels = model.predict(X)
    probabilities = model.predict_proba(X)

    print("GMM completed.")
    print(f"Converged: {model.converged_}")
    print(f"Iterations: {model.n_iter_}")
    print(f"Lower bound: {model.lower_bound_:.4f}")

    if not model.converged_:
        print(
            "WARNING: GMM did not converge. "
            "Consider increasing max_iter."
        )

    return model, labels, probabilities


# ---------------------------------------------------------
# Clustering evaluation
# ---------------------------------------------------------

def evaluate_clustering(
    X,
    labels,
    true_categories,
    algorithm,
    sample_size=5000,
    random_state=42,
):
    """
    Calculate:
        - Silhouette score
        - Davies-Bouldin index
        - Calinski-Harabasz index
        - Normalized Mutual Information

    Internal metrics use a reproducible sample.
    NMI uses all article assignments.
    """

    labels = np.asarray(labels)
    true_categories = np.asarray(true_categories)

    if len(labels) != len(X):
        raise ValueError(
            f"{algorithm}: labels and feature matrix have different lengths."
        )

    unique_clusters = np.unique(labels)

    if len(unique_clusters) < 2:
        raise ValueError(
            f"{algorithm} produced fewer than two clusters."
        )

    if len(unique_clusters) >= len(labels):
        raise ValueError(
            f"{algorithm} produced one cluster per article."
        )

    rng = np.random.default_rng(random_state)

    n = min(sample_size, len(X))

    positions = rng.choice(
        len(X),
        size=n,
        replace=False,
    )

    X_sample = X[positions]
    labels_sample = labels[positions]

    # Ensure the sample contains at least two clusters.
    if len(np.unique(labels_sample)) < 2:
        raise ValueError(
            f"Evaluation sample contains fewer than two clusters for {algorithm}. "
            "Increase evaluation_sample_size."
        )

    if n <= len(np.unique(labels_sample)):
        raise ValueError(
            f"Not enough evaluation samples for {algorithm}."
        )

    print(
        f"\nEvaluating {algorithm} on {n:,} sampled articles..."
    )

    silhouette = silhouette_score(
        X_sample,
        labels_sample,
        metric="euclidean",
    )

    davies_bouldin = davies_bouldin_score(
        X_sample,
        labels_sample,
    )

    calinski_harabasz = calinski_harabasz_score(
        X_sample,
        labels_sample,
    )

    nmi = normalized_mutual_info_score(
        true_categories,
        labels,
    )

    result = {
        "algorithm": algorithm,
        "n_clusters": int(len(unique_clusters)),
        "silhouette_score": float(silhouette),
        "davies_bouldin_score": float(davies_bouldin),
        "calinski_harabasz_score": float(calinski_harabasz),
        "nmi_vs_category": float(nmi),
        "evaluation_sample_size": int(n),
    }

    print("\nEvaluation results:")

    for metric, value in result.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

    return result


# ---------------------------------------------------------
# Topic interpretation
# ---------------------------------------------------------

def extract_topic_keywords(
    X_tfidf,
    labels,
    vectorizer,
    algorithm,
    top_n=15,
):
    """
    Extract representative TF-IDF keywords for every cluster.

    Terms are ranked by mean TF-IDF score within the cluster.
    """

    feature_names = vectorizer.get_feature_names_out()
    records = []

    for cluster_id in np.unique(labels):

        positions = np.flatnonzero(
            labels == cluster_id
        )

        cluster_matrix = X_tfidf[positions]

        mean_scores = np.asarray(
            cluster_matrix.mean(axis=0)
        ).ravel()

        top_indices = np.argsort(
            mean_scores
        )[-top_n:][::-1]

        for rank, index in enumerate(
            top_indices,
            start=1,
        ):

            score = float(mean_scores[index])

            if score <= 0:
                continue

            records.append({
                "algorithm": algorithm,
                "cluster": int(cluster_id),
                "rank": rank,
                "term": str(feature_names[index]),
                "mean_tfidf": score,
                "cluster_size": int(len(positions)),
            })

    return records


def extract_topic_examples(
    df,
    X,
    labels,
    model,
    algorithm,
    examples_per_topic=5,
):
    """
    Select articles closest to each cluster/component center.
    """

    if algorithm == "KMeans":
        centers = model.cluster_centers_

    elif algorithm == "GMM":
        centers = model.means_

    else:
        raise ValueError(
            f"Unknown algorithm: {algorithm}"
        )

    records = []

    for cluster_id in np.unique(labels):

        positions = np.flatnonzero(
            labels == cluster_id
        )

        center = centers[cluster_id]

        distances = np.linalg.norm(
            X[positions] - center,
            axis=1,
        )

        nearest_order = np.argsort(
            distances
        )[:examples_per_topic]

        for rank, local_position in enumerate(
            nearest_order,
            start=1,
        ):

            article_position = positions[local_position]
            article = df.iloc[article_position]

            headline = (
                article["headline"]
                if "headline" in df.columns
                else str(article["text"])[:200]
            )

            records.append({
                "algorithm": algorithm,
                "cluster": int(cluster_id),
                "rank": rank,
                "id": int(article["id"]),
                "category": article["category"],
                "headline": headline,
                "text": str(article["text"])[:500],
                "distance_to_center": float(
                    distances[local_position]
                ),
            })

    return records


# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------

def run(
    n_clusters=8,
    evaluation_sample_size=5000,
    top_n_keywords=15,
    examples_per_topic=5,
    random_state=42,
):

    if evaluation_sample_size < 2:
        raise ValueError(
            "evaluation_sample_size must be at least 2."
        )

    if top_n_keywords < 1:
        raise ValueError(
            "top_n_keywords must be at least 1."
        )

    if examples_per_topic < 1:
        raise ValueError(
            "examples_per_topic must be at least 1."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df, X, X_tfidf, vectorizer = load_data()

    if n_clusters < 2 or n_clusters >= len(df):
        raise ValueError(
            "n_clusters must be at least 2 and smaller "
            "than the number of articles."
        )

    # -----------------------------------------------------
    # K-Means
    # -----------------------------------------------------

    kmeans_model, kmeans_labels = run_kmeans(
        X,
        n_clusters=n_clusters,
        random_state=random_state,
    )

    # -----------------------------------------------------
    # GMM
    # -----------------------------------------------------

    gmm_model, gmm_labels, gmm_probabilities = run_gmm(
        X,
        n_components=n_clusters,
        random_state=random_state,
    )

    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------

    metrics = []

    metrics.append(
        evaluate_clustering(
            X,
            kmeans_labels,
            df["category"].to_numpy(),
            algorithm="KMeans",
            sample_size=evaluation_sample_size,
            random_state=random_state,
        )
    )

    metrics.append(
        evaluate_clustering(
            X,
            gmm_labels,
            df["category"].to_numpy(),
            algorithm="GMM",
            sample_size=evaluation_sample_size,
            random_state=random_state,
        )
    )

    # -----------------------------------------------------
    # Article assignments
    # -----------------------------------------------------

    assignment_columns = [
        column
        for column in [
            "id",
            "date",
            "month",
            "category",
            "headline",
        ]
        if column in df.columns
    ]

    assignments = df[assignment_columns].copy()

    assignments["kmeans_cluster"] = kmeans_labels
    assignments["gmm_cluster"] = gmm_labels

    assignments["gmm_confidence"] = (
        gmm_probabilities.max(axis=1)
    )

    assignments.to_parquet(
        ASSIGNMENTS_PATH,
        index=False,
    )

    # -----------------------------------------------------
    # Save models and probabilities
    # -----------------------------------------------------

    joblib.dump(
        kmeans_model,
        KMEANS_PATH,
    )

    joblib.dump(
        gmm_model,
        GMM_PATH,
    )

    np.save(
        GMM_PROBABILITIES_PATH,
        gmm_probabilities.astype(np.float32),
    )

    # -----------------------------------------------------
    # Topic keywords and representative articles
    # -----------------------------------------------------

    keyword_records = []
    example_records = []

    algorithms = [
        ("KMeans", kmeans_labels, kmeans_model),
        ("GMM", gmm_labels, gmm_model),
    ]

    for algorithm, labels, model in algorithms:

        keyword_records.extend(
            extract_topic_keywords(
                X_tfidf,
                labels,
                vectorizer,
                algorithm,
                top_n=top_n_keywords,
            )
        )

        example_records.extend(
            extract_topic_examples(
                df,
                X,
                labels,
                model,
                algorithm,
                examples_per_topic=examples_per_topic,
            )
        )

    # -----------------------------------------------------
    # Save reports
    # -----------------------------------------------------

    pd.DataFrame(metrics).to_csv(
        METRICS_PATH,
        index=False,
    )

    pd.DataFrame(keyword_records).to_csv(
        KEYWORDS_PATH,
        index=False,
    )

    pd.DataFrame(example_records).to_csv(
        EXAMPLES_PATH,
        index=False,
    )

    # -----------------------------------------------------
    # Completion
    # -----------------------------------------------------

    print("\n" + "=" * 55)
    print("TOPIC DISCOVERY COMPLETED")
    print("=" * 55)

    print(f"Article assignments: {ASSIGNMENTS_PATH}")
    print(f"Clustering metrics: {METRICS_PATH}")
    print(f"Topic keywords: {KEYWORDS_PATH}")
    print(f"Topic examples: {EXAMPLES_PATH}")
    print(f"K-Means model: {KMEANS_PATH}")
    print(f"GMM model: {GMM_PATH}")
    print(f"GMM probabilities: {GMM_PROBABILITIES_PATH}")


# ---------------------------------------------------------
# Command-line execution
# ---------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Discover news topics using K-Means and GMM."
    )

    parser.add_argument(
        "--clusters",
        type=int,
        default=8,
        help="Number of clusters/components.",
    )

    parser.add_argument(
        "--evaluation-sample-size",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--top-keywords",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--examples-per-topic",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    run(
        n_clusters=args.clusters,
        evaluation_sample_size=args.evaluation_sample_size,
        top_n_keywords=args.top_keywords,
        examples_per_topic=args.examples_per_topic,
        random_state=args.random_state,
    )
