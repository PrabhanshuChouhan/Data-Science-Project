
"""
Text representation module.

Creates:
1. Dense semantic embeddings using Sentence Transformers.
2. Sparse TF-IDF vectors using scikit-learn.

Input:
    data/processed/news.parquet

Outputs:
    data/processed/emb.npy
    data/processed/tfidf.npz
    data/processed/tfidf_vectorizer.joblib
    data/processed/representation_metadata.json

Important:
    Every output is aligned to the original article `id`.
    Therefore, embeddings can be retrieved using E_all[article_id].
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------------------------------------
# Paths and defaults
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

DATA_PATH = PROCESSED_DIR / "news.parquet"
EMBEDDINGS_PATH = PROCESSED_DIR / "emb.npy"
TFIDF_PATH = PROCESSED_DIR / "tfidf.npz"
VECTORIZER_PATH = PROCESSED_DIR / "tfidf_vectorizer.joblib"
METADATA_PATH = PROCESSED_DIR / "representation_metadata.json"

DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"


# ---------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------

def load_articles(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load articles and enforce the article-ID alignment invariant."""

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_parquet(path)

    required = {"id", "text"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    if df["id"].isna().any():
        raise ValueError("Article IDs contain missing values.")

    if df["id"].duplicated().any():
        raise ValueError("Article IDs must be unique.")

    if not pd.api.types.is_integer_dtype(df["id"]):
        raise TypeError("Article IDs must be integers.")

    if (df["id"] < 0).any():
        raise ValueError("Article IDs cannot be negative.")

    # This is necessary because downstream modules use:
    # embeddings[article_id]
    df = df.sort_values("id").reset_index(drop=True)

    expected_ids = np.arange(len(df))

    if not np.array_equal(df["id"].to_numpy(), expected_ids):
        raise ValueError(
            "Article IDs must be contiguous from 0 to N-1. "
            "Do not reset or regenerate IDs after cleaning."
        )

    df["text"] = df["text"].fillna("").astype(str).str.strip()

    if (df["text"].str.len() == 0).any():
        raise ValueError("Some articles have empty text.")

    print(f"Loaded {len(df):,} articles.")
    print("Article IDs are valid and correctly aligned.")

    return df


# ---------------------------------------------------------
# Dense semantic embeddings
# ---------------------------------------------------------

def generate_embeddings(
    texts: list[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
    device: str | None = None,
) -> np.ndarray:
    """
    Generate normalized sentence embeddings.

    Normalization makes cosine similarity equivalent to
    dot-product similarity for these vectors.
    """

    if not texts:
        raise ValueError("Cannot generate embeddings for an empty dataset.")

    print(f"\nLoading embedding model: {model_name}")

    model = SentenceTransformer(
        model_name,
        device=device,
    )

    print(f"Generating embeddings for {len(texts):,} articles...")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2:
        raise ValueError("Embedding output must be a 2D matrix.")

    if embeddings.shape[0] != len(texts):
        raise ValueError("Embedding count does not match article count.")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or infinite values.")

    print(f"Embedding matrix shape: {embeddings.shape}")

    return embeddings


# ---------------------------------------------------------
# Sparse TF-IDF representation
# ---------------------------------------------------------

def generate_tfidf(
    texts: list[str],
    max_features: int = 50_000,
):
    """
    Generate sparse TF-IDF vectors using unigrams and bigrams.

    Returns:
        matrix: Sparse document-term matrix.
        vectorizer: Fitted TfidfVectorizer.
    """

    print("\nGenerating TF-IDF representation...")

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words="english",
        sublinear_tf=True,
        dtype=np.float32,
    )

    matrix = vectorizer.fit_transform(texts)

    if matrix.shape[0] != len(texts):
        raise ValueError("TF-IDF row count does not match article count.")

    print(f"TF-IDF matrix shape: {matrix.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_):,}")

    return matrix, vectorizer


# ---------------------------------------------------------
# Save outputs
# ---------------------------------------------------------

def save_metadata(
    df: pd.DataFrame,
    embeddings: np.ndarray | None,
    tfidf,
    model_name: str,
) -> None:

    metadata = {
        "number_of_articles": int(len(df)),
        "id_column": "id",
        "id_alignment": "row_index_equals_article_id",
        "embedding_model": model_name if embeddings is not None else None,
        "embedding_shape": (
            list(embeddings.shape) if embeddings is not None else None
        ),
        "tfidf_shape": list(tfidf.shape) if tfidf is not None else None,
        "tfidf_description": (
            "TF-IDF with unigrams and bigrams"
            if tfidf is not None else None
        ),
    }

    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)


# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------

def run(
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
    device: str | None = None,
    max_features: int = 50_000,
    force: bool = False,
) -> None:

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = load_articles()

    texts = df["text"].tolist()

    embeddings = None
    tfidf = None

    # Dense embeddings
    if EMBEDDINGS_PATH.exists() and not force:
        print(f"\nEmbeddings already exist: {EMBEDDINGS_PATH}")
        print("Skipping generation. Use --force to regenerate.")
    else:
        embeddings = generate_embeddings(
            texts=texts,
            model_name=model_name,
            batch_size=batch_size,
            device=device,
        )

        np.save(EMBEDDINGS_PATH, embeddings)
        print(f"Saved embeddings: {EMBEDDINGS_PATH}")

    # Sparse TF-IDF vectors
    if TFIDF_PATH.exists() and VECTORIZER_PATH.exists() and not force:
        print(f"\nTF-IDF already exists: {TFIDF_PATH}")
        print("Skipping generation. Use --force to regenerate.")
    else:
        tfidf, vectorizer = generate_tfidf(
            texts=texts,
            max_features=max_features,
        )

        save_npz(TFIDF_PATH, tfidf)
        joblib.dump(vectorizer, VECTORIZER_PATH)

        print(f"Saved TF-IDF matrix: {TFIDF_PATH}")
        print(f"Saved vectorizer: {VECTORIZER_PATH}")

    save_metadata(
        df=df,
        embeddings=embeddings,
        tfidf=tfidf,
        model_name=model_name,
    )

    print("\nRepresentation pipeline completed.")
    print(f"Articles processed: {len(df):,}")


# ---------------------------------------------------------
# Command-line execution
# ---------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate dense and sparse news representations."
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Sentence Transformer model name.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device: cpu, cuda, or another supported device.",
    )

    parser.add_argument(
        "--max-features",
        type=int,
        default=50_000,
        help="Maximum TF-IDF vocabulary size.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate existing representation files.",
    )

    args = parser.parse_args()

    run(
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
        max_features=args.max_features,
        force=args.force,
    )
