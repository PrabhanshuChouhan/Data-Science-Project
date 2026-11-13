
"""
Dimensionality reduction module.

Input:
    data/processed/news.parquet
    data/processed/emb.npy

Outputs:
    data/processed/pca_embeddings.npy
    data/processed/pca_model.joblib
    data/processed/tsne_2d.npy
    data/processed/tsne_ids.npy
    data/processed/reduction_metadata.json

PCA:
    Applied to all article embeddings.

t-SNE:
    Applied to a reproducible sample of PCA-reduced embeddings.
    Used for 2D visualization, not downstream model training.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ---------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

DATA_PATH = PROCESSED_DIR / "news.parquet"
EMBEDDINGS_PATH = PROCESSED_DIR / "emb.npy"

PCA_OUTPUT_PATH = PROCESSED_DIR / "pca_embeddings.npy"
PCA_MODEL_PATH = PROCESSED_DIR / "pca_model.joblib"

TSNE_OUTPUT_PATH = PROCESSED_DIR / "tsne_2d.npy"
TSNE_IDS_PATH = PROCESSED_DIR / "tsne_ids.npy"

METADATA_PATH = PROCESSED_DIR / "reduction_metadata.json"

# ---------------------------------------------------------
# Load and validate data
# ---------------------------------------------------------

def load_data():
    """Load articles and embeddings while preserving ID alignment."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Embeddings not found: {EMBEDDINGS_PATH}. "
            "Run represent.py first."
        )

    df = pd.read_parquet(DATA_PATH)
    embeddings = np.load(EMBEDDINGS_PATH, mmap_mode="r")

    if "id" not in df.columns:
        raise ValueError("Dataset must contain an 'id' column.")

    if df["id"].duplicated().any():
        raise ValueError("Article IDs must be unique.")

    df = df.sort_values("id").reset_index(drop=True)

    expected_ids = np.arange(len(df))

    if not np.array_equal(df["id"].to_numpy(), expected_ids):
        raise ValueError(
            "Article IDs must be contiguous from 0 to N-1."
        )

    if embeddings.ndim != 2:
        raise ValueError("Embedding matrix must be two-dimensional.")

    if embeddings.shape[0] != len(df):
        raise ValueError(
            "Article count and embedding count do not match."
        )

    print(f"Articles loaded: {len(df):,}")
    print(f"Original embedding shape: {embeddings.shape}")

    return df, embeddings

# ---------------------------------------------------------
# PCA
# ---------------------------------------------------------

def reduce_with_pca(
    embeddings,
    n_components=50,
    random_state=42,
):
    """
    Reduce all embeddings using scikit-learn PCA.

    Returns:
        reduced: PCA representation.
        model: fitted PCA model.
    """

    n_samples, n_features = embeddings.shape

    if n_components <= 0:
        raise ValueError("n_components must be positive.")

    if n_components > min(n_samples, n_features):
        raise ValueError(
            f"n_components cannot exceed {min(n_samples, n_features)}."
        )

    print(f"\nRunning PCA: {n_features} -> {n_components} dimensions")

    model = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=random_state,
    )

    reduced = model.fit_transform(embeddings)
    reduced = np.asarray(reduced, dtype=np.float32)

    if not np.isfinite(reduced).all():
        raise ValueError("PCA output contains invalid values.")

    explained = float(model.explained_variance_ratio_.sum())

    print(f"PCA output shape: {reduced.shape}")
    print(f"Total explained variance: {explained:.4f}")

    return reduced, model

# ---------------------------------------------------------
# t-SNE
# ---------------------------------------------------------

def reduce_with_tsne(
    pca_embeddings,
    article_ids,
    sample_size=5000,
    perplexity=30,
    random_state=42,
):
    """
    Generate a 2D t-SNE representation from a reproducible sample.

    t-SNE is intentionally not run on the entire dataset because
    its computational cost grows significantly with sample size.

    Returns:
        coordinates: 2D t-SNE coordinates.
        sampled_ids: original article IDs corresponding to coordinates.
    """

    n_samples = len(pca_embeddings)

    if sample_size < 2:
        raise ValueError("t-SNE requires at least two samples.")

    actual_size = min(sample_size, n_samples)

    if perplexity <= 0 or perplexity >= actual_size:
        raise ValueError(
            "Perplexity must be positive and smaller than the sample size."
        )

    rng = np.random.default_rng(random_state)

    selected_positions = np.sort(
        rng.choice(
            n_samples,
            size=actual_size,
            replace=False,
        )
    )

    sample = pca_embeddings[selected_positions]
    sampled_ids = article_ids[selected_positions]

    print(
        f"\nRunning t-SNE on {actual_size:,} articles "
        f"with perplexity={perplexity}..."
    )

    model = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        random_state=random_state,
    )

    coordinates = model.fit_transform(sample)
    coordinates = np.asarray(coordinates, dtype=np.float32)

    if not np.isfinite(coordinates).all():
        raise ValueError("t-SNE output contains invalid values.")

    print(f"t-SNE output shape: {coordinates.shape}")

    return coordinates, sampled_ids

# ---------------------------------------------------------
# Save metadata
# ---------------------------------------------------------

def save_metadata(
    original_shape,
    pca_shape,
    tsne_shape,
    pca_model,
    sample_size,
    perplexity,
):
    metadata = {
        "original_embedding_shape": list(original_shape),
        "pca_shape": list(pca_shape),
        "pca_components": int(pca_shape[1]),
        "pca_explained_variance_ratio": float(
            pca_model.explained_variance_ratio_.sum()
        ),
        "tsne_shape": list(tsne_shape),
        "tsne_sample_size": int(sample_size),
        "tsne_perplexity": float(perplexity),
        "random_state": 42,
        "id_alignment": "row_index_equals_article_id",
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------

def run(
    n_components=50,
    tsne_sample_size=5000,
    perplexity=30,
    random_state=42,
    force=False,
):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df, embeddings = load_data()
    article_ids = df["id"].to_numpy()

    # PCA
    if (
        PCA_OUTPUT_PATH.exists()
        and PCA_MODEL_PATH.exists()
        and not force
    ):
        print("\nExisting PCA outputs found. Loading them.")

        pca_embeddings = np.load(PCA_OUTPUT_PATH, mmap_mode="r")
        pca_model = joblib.load(PCA_MODEL_PATH)

        if pca_embeddings.shape[0] != len(df):
            raise ValueError("Existing PCA output has the wrong row count.")

        if pca_embeddings.shape[1] != n_components:
            raise ValueError(
                "Existing PCA output has a different component count. "
                "Run with --force."
            )

    else:
        pca_embeddings, pca_model = reduce_with_pca(
            embeddings=embeddings,
            n_components=n_components,
            random_state=random_state,
        )

        np.save(PCA_OUTPUT_PATH, pca_embeddings)
        joblib.dump(pca_model, PCA_MODEL_PATH)

        print(f"Saved PCA embeddings: {PCA_OUTPUT_PATH}")
        print(f"Saved PCA model: {PCA_MODEL_PATH}")

    # t-SNE
    if (
        TSNE_OUTPUT_PATH.exists()
        and TSNE_IDS_PATH.exists()
        and not force
    ):
        print("\nExisting t-SNE outputs found. Loading them.")

        tsne_coordinates = np.load(TSNE_OUTPUT_PATH)
        tsne_ids = np.load(TSNE_IDS_PATH)

        if len(tsne_coordinates) != len(tsne_ids):
            raise ValueError("t-SNE coordinates and IDs do not match.")

    else:
        tsne_coordinates, tsne_ids = reduce_with_tsne(
            pca_embeddings=pca_embeddings,
            article_ids=article_ids,
            sample_size=tsne_sample_size,
            perplexity=perplexity,
            random_state=random_state,
        )

        np.save(TSNE_OUTPUT_PATH, tsne_coordinates)
        np.save(TSNE_IDS_PATH, tsne_ids)

        print(f"Saved t-SNE coordinates: {TSNE_OUTPUT_PATH}")
        print(f"Saved t-SNE article IDs: {TSNE_IDS_PATH}")

    save_metadata(
        original_shape=embeddings.shape,
        pca_shape=pca_embeddings.shape,
        tsne_shape=tsne_coordinates.shape,
        pca_model=pca_model,
        sample_size=len(tsne_coordinates),
        perplexity=perplexity,
    )

    print("\nDimensionality reduction completed.")
    print(f"Original dimensions: {embeddings.shape[1]}")
    print(f"PCA dimensions: {pca_embeddings.shape[1]}")
    print(f"t-SNE dimensions: {tsne_coordinates.shape[1]}")

# ---------------------------------------------------------
# Command-line execution
# ---------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Perform PCA and t-SNE on news embeddings."
    )

    parser.add_argument(
        "--components",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--tsne-sample-size",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--perplexity",
        type=float,
        default=30,
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    run(
        n_components=args.components,
        tsne_sample_size=args.tsne_sample_size,
        perplexity=args.perplexity,
        random_state=args.random_state,
        force=args.force,
    )