from pathlib import Path
import json
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.neighbors import NearestNeighbors

BASE_DIR = Path.cwd()

NEWS_PATH = BASE_DIR / "data" / "processed" / "news.parquet"
EMBEDDING_PATH = BASE_DIR / "data" / "processed" / "emb.npy"

REPORT_DIR = BASE_DIR / "results" / "streaming" / "reports"
OUTPUT_DIR = BASE_DIR / "results" / "streaming" / "analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():

    print("Loading article data...")

    news = pd.read_parquet(NEWS_PATH)
    embeddings = np.load(EMBEDDING_PATH)

    news["date"] = pd.to_datetime(news["date"])

    ids = news["id"].to_numpy(dtype=int)

    if len(embeddings) <= ids.max():
        raise ValueError("Embedding array does not cover all article IDs.")

    embeddings = embeddings[ids]

    if len(news) != len(embeddings):
        raise ValueError("News and embedding counts do not match.")

    return news, embeddings

def select_articles(news, embeddings, max_per_month=30):

    print("Selecting representative articles...")

    sampled_indices = []

    for month, group in news.groupby("month", sort=True):

        sample = group.sample(
            n=min(max_per_month, len(group)),
            random_state=42
        )

        sampled_indices.extend(sample.index.tolist())

    sampled_news = news.loc[sampled_indices].copy()

    # Embeddings are aligned to the original news row order.
    positions = news.index.get_indexer(sampled_news.index)
    sampled_embeddings = embeddings[positions]

    sampled_news = sampled_news.reset_index(drop=True)

    print(f"Selected {len(sampled_news)} articles.")

    return sampled_news, sampled_embeddings

def build_similarity_graph(news, embeddings, k=6, threshold=0.55):

    print("Finding similar articles...")

    model = NearestNeighbors(
        n_neighbors=min(k + 1, len(news)),
        metric="cosine",
        algorithm="brute",
        n_jobs=-1
    )

    model.fit(embeddings)

    distances, indices = model.kneighbors(embeddings)

    graph = nx.Graph()

    # Add article nodes.
    for i, article in news.iterrows():

        graph.add_node(
            int(article["id"]),
            date=article["date"].strftime("%Y-%m-%d"),
            month=str(article["month"]),
            category=str(article["category"]),
            text_preview=str(article["text"])[:250]
        )

    # Add similarity edges.
    for i in range(len(news)):

        source_id = int(news.iloc[i]["id"])

        for distance, neighbor_index in zip(
            distances[i][1:],
            indices[i][1:]
        ):

            similarity = 1.0 - float(distance)

            if similarity < threshold:
                continue

            target_id = int(news.iloc[neighbor_index]["id"])

            if source_id == target_id:
                continue

            graph.add_edge(
                source_id,
                target_id,
                weight=similarity
            )

    print(f"Graph created: {graph.number_of_nodes()} nodes, "
          f"{graph.number_of_edges()} edges.")

    return graph


def calculate_metrics(graph):

    print("Calculating PageRank...")

    pagerank = nx.pagerank(
        graph,
        weight="weight"
    )

    degree = dict(graph.degree())

    return pagerank, degree


def export_graph(graph, pagerank, degree):

    print("Exporting graph data...")

    nodes = []

    for node_id, attributes in graph.nodes(data=True):

        nodes.append({
            "id": int(node_id),
            "date": attributes["date"],
            "month": attributes["month"],
            "category": attributes["category"],
            "text_preview": attributes["text_preview"],
            "pagerank": float(pagerank[node_id]),
            "degree": int(degree[node_id])
        })

    edges = []

    for source, target, attributes in graph.edges(data=True):

        edges.append({
            "source": int(source),
            "target": int(target),
            "similarity": float(attributes["weight"])
        })

    nodes.sort(
        key=lambda x: x["pagerank"],
        reverse=True
    )

    output = {
        "metadata": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "similarity_threshold": 0.55
        },
        "nodes": nodes,
        "edges": edges
    }

    output_path = OUTPUT_DIR / "article_network.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False)

    # Also save the most influential articles separately.
    top_articles = pd.DataFrame(nodes).head(20)

    top_articles.to_csv(
        OUTPUT_DIR / "top_influential_articles.csv",
        index=False
    )

    print(f"Graph saved to: {output_path}")
    print("Top influential articles saved.")

def run_article_network():

    news, embeddings = load_data()

    sampled_news, sampled_embeddings = select_articles(
        news,
        embeddings
    )

    graph = build_similarity_graph(
        sampled_news,
        sampled_embeddings
    )

    pagerank, degree = calculate_metrics(graph)

    export_graph(
        graph,
        pagerank,
        degree
    )

    print("\nArticle network generation completed.")


if __name__ == "__main__":
    run_article_network()
