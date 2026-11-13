
"""
Article relationship and influence analysis.

Uses:
    FAISS       - Similarity search
    NetworkX    - Graph construction and analysis
    Louvain     - Community detection
    PageRank    - Article influence ranking

Inputs:
    data/processed/news.parquet
    data/processed/emb.npy

Outputs:
    results/graph/news_graph.graphml
    results/graph/article_graph_metrics.parquet
    results/graph/community_summary.csv
    results/graph/graph_metadata.json
"""

import argparse
import json
from pathlib import Path

import faiss
import networkx as nx
import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "results" / "graph"

DATA_PATH = PROCESSED_DIR / "news.parquet"
EMBEDDINGS_PATH = PROCESSED_DIR / "emb.npy"

GRAPH_PATH = OUTPUT_DIR / "news_graph.graphml"
METRICS_PATH = OUTPUT_DIR / "article_graph_metrics.parquet"
COMMUNITY_PATH = OUTPUT_DIR / "community_summary.csv"
METADATA_PATH = OUTPUT_DIR / "graph_metadata.json"


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(EMBEDDINGS_PATH)

    df = pd.read_parquet(DATA_PATH)
    embeddings = np.load(
        EMBEDDINGS_PATH,
        mmap_mode="r",
    )

    required = {"id", "category", "text"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing dataset columns: {missing}"
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

    if embeddings.ndim != 2:
        raise ValueError(
            "Embeddings must be a 2D matrix."
        )

    if embeddings.shape[0] != len(df):
        raise ValueError(
            "Embedding rows do not match article count."
        )

    print(f"Articles: {len(df):,}")
    print(f"Embedding dimensions: {embeddings.shape[1]}")

    return df, np.asarray(embeddings, dtype=np.float32)


# ---------------------------------------------------------
# Build similarity graph
# ---------------------------------------------------------

def build_similarity_graph(
    embeddings,
    top_k=10,
    threshold=0.65,
):
    """
    Build an undirected article similarity graph.

    Each article is connected to its top-k most similar
    articles if cosine similarity >= threshold.

    FAISS performs the similarity search.
    """

    print("\nNormalizing embeddings...")

    X = normalize(
        embeddings,
        norm="l2",
        axis=1,
    ).astype(np.float32)

    n_articles, dimensions = X.shape

    if top_k < 1 or top_k >= n_articles:
        raise ValueError(
            "top_k must be at least 1 and smaller than article count."
        )

    print("\nBuilding FAISS similarity index...")

    index = faiss.IndexFlatIP(dimensions)
    index.add(X)

    print("Searching for similar articles...")

    similarities, neighbors = index.search(
        X,
        top_k + 1,
    )

    graph = nx.Graph()

    graph.add_nodes_from(range(n_articles))

    print("Creating graph edges...")

    for article_id in range(n_articles):

        for similarity, neighbor_id in zip(
            similarities[article_id],
            neighbors[article_id],
        ):

            neighbor_id = int(neighbor_id)

            if neighbor_id == article_id:
                continue

            similarity = float(similarity)

            if similarity < threshold:
                continue

            graph.add_edge(
                article_id,
                neighbor_id,
                weight=similarity,
            )

    print("\nGraph construction completed.")
    print(f"Nodes: {graph.number_of_nodes():,}")
    print(f"Edges: {graph.number_of_edges():,}")

    return graph


# ---------------------------------------------------------
# Graph analysis
# ---------------------------------------------------------

def calculate_graph_metrics(graph):

    print("\nCalculating graph metrics...")

    print("Calculating PageRank...")

    pagerank = nx.pagerank(
        graph,
        alpha=0.85,
        weight="weight",
        max_iter=100,
    )

    print("Calculating degree centrality...")

    degree_centrality = nx.degree_centrality(graph)

    print("Calculating weighted degree...")

    weighted_degree = dict(
        graph.degree(weight="weight")
    )

    print("Detecting communities...")

    if graph.number_of_edges() > 0:
        communities = nx.community.louvain_communities(
            graph,
            weight="weight",
            seed=42,
        )
    else:
        communities = [
            {node}
            for node in graph.nodes()
        ]

    community_assignment = {}

    for community_id, community in enumerate(communities):
        for node in community:
            community_assignment[node] = community_id

    metrics = pd.DataFrame({
        "id": list(graph.nodes()),
        "pagerank": [
            pagerank[node]
            for node in graph.nodes()
        ],
        "degree_centrality": [
            degree_centrality[node]
            for node in graph.nodes()
        ],
        "weighted_degree": [
            weighted_degree[node]
            for node in graph.nodes()
        ],
        "community": [
            community_assignment[node]
            for node in graph.nodes()
        ],
    })

    print(
        f"Communities detected: {len(communities):,}"
    )

    return metrics, communities


# ---------------------------------------------------------
# Community interpretation
# ---------------------------------------------------------

def summarize_communities(
    df,
    metrics,
    communities,
):

    print("\nSummarizing communities...")

    merged = df.merge(
        metrics,
        on="id",
        how="left",
        validate="one_to_one",
    )

    records = []

    for community_id, group in merged.groupby("community"):

        category_counts = (
            group["category"]
            .value_counts()
        )

        dominant_category = (
            category_counts.index[0]
            if len(category_counts) > 0
            else "Unknown"
        )

        top_articles = (
            group.nlargest(5, "pagerank")
        )

        records.append({
            "community": int(community_id),
            "article_count": int(len(group)),
            "dominant_category": dominant_category,
            "category_distribution": json.dumps(
                category_counts.to_dict()
            ),
            "average_pagerank": float(
                group["pagerank"].mean()
            ),
            "top_article_ids": json.dumps(
                top_articles["id"].astype(int).tolist()
            ),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------

def run(
    top_k=10,
    threshold=0.65,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df, embeddings = load_data()

    graph = build_similarity_graph(
        embeddings,
        top_k=top_k,
        threshold=threshold,
    )

    metrics, communities = calculate_graph_metrics(
        graph
    )

    # Save article-level graph metrics.
    metrics.to_parquet(
        METRICS_PATH,
        index=False,
    )

    # Save community summaries.
    community_summary = summarize_communities(
        df,
        metrics,
        communities,
    )

    community_summary.to_csv(
        COMMUNITY_PATH,
        index=False,
    )

    # Save graph.
    nx.write_graphml(
        graph,
        GRAPH_PATH,
    )

    # Save metadata.
    metadata = {
        "number_of_articles": int(
            graph.number_of_nodes()
        ),
        "number_of_edges": int(
            graph.number_of_edges()
        ),
        "number_of_communities": int(
            len(communities)
        ),
        "top_k": int(top_k),
        "similarity_threshold": float(threshold),
        "similarity_metric": "cosine_similarity",
        "community_algorithm": "Louvain",
        "influence_algorithm": "PageRank",
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    print("\n" + "=" * 55)
    print("ARTICLE GRAPH ANALYSIS COMPLETED")
    print("=" * 55)

    print(f"Graph: {GRAPH_PATH}")
    print(f"Article metrics: {METRICS_PATH}")
    print(f"Community summary: {COMMUNITY_PATH}")
    print(f"Metadata: {METADATA_PATH}")


# ---------------------------------------------------------
# Command-line execution
# ---------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Analyze article relationships and influence."
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of nearest neighbors per article.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Minimum cosine similarity for graph edges.",
    )

    args = parser.parse_args()

    run(
        top_k=args.top_k,
        threshold=args.threshold,
    )
