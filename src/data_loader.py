from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

NEWS_PATH = DATA_DIR / "news.parquet"
EMBEDDINGS_PATH = DATA_DIR / "emb.npy"
PCA_PATH = DATA_DIR / "pca_embeddings.npy"

def load_news():
    if not NEWS_PATH.exists():
        raise FileNotFoundError(f"News data not found at {NEWS_PATH}. Please run the cleaning script first.")
    df = pd.read_parquet(NEWS_PATH)
    
    required_columns = ["id", "date", "month", "category", "authors", "text"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"News data is missing required columns. Expected columns: {required_columns}")
    
    df=df.sort_values("date").reset_index(drop=True)
    return df

def load_embeddings():
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"Embeddings not found at {EMBEDDINGS_PATH}. Please run the embedding generation script first.")
    embeddings = np.load(EMBEDDINGS_PATH)
    return embeddings

def load_pca_embeddings():
    if not PCA_PATH.exists():
        raise FileNotFoundError(f"PCA embeddings not found at {PCA_PATH}. Please run the PCA generation script first.")
    pca_embeddings = np.load(PCA_PATH)
    return pca_embeddings

def load_all():
    df = load_news()
    embeddings = load_embeddings()
    pca_embeddings = load_pca_embeddings()
    n_articles = len(df)
    if embeddings.shape[0] != n_articles:
        raise ValueError(f"Number of embeddings ({embeddings.shape[0]}) does not match number of articles ({n_articles}).")
    
    if pca_embeddings.shape[0] != n_articles:
        raise ValueError(f"Number of PCA embeddings ({pca_embeddings.shape[0]}) does not match number of articles ({n_articles}).")
    
    return df, embeddings, pca_embeddings

def main():
    df, embeddings, pca_embeddings = load_all()
    print(f"Loaded {len(df)} articles with embeddings of shape {embeddings.shape} and PCA embeddings of shape {pca_embeddings.shape}.")
    
if __name__ == "__main__":
    main()