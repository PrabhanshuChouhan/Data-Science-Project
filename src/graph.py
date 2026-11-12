import itertools
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_parquet("data/processed/news.parquet")
Z = np.load("data/processed/emb_pca50.npy")

# 1. Author Co-Authorship Graph
G_auth = nx.Graph()
for authors_str in df["authors"].dropna():
    names = [a.strip() for a in str(authors_str).split(",") if a.strip() and "contributor" not in a.lower()]
    if len(names) == 1:
        G_auth.add_node(names[0])
    elif len(names) > 1:
        for u, v in itertools.combinations(sorted(set(names)), 2):
            if G_auth.has_edge(u, v):
                G_auth[u][v]["weight"] += 1
            else:
                G_auth.add_edge(u, v, weight=1)

print(f"Author Graph -> Nodes: {G_auth.number_of_nodes()} | Edges: {G_auth.number_of_edges()}")

if G_auth.number_of_edges() > 0:
    lcc_nodes = max(nx.connected_components(G_auth), key=len)
    G_lcc = G_auth.subgraph(lcc_nodes).copy()
    pr = nx.pagerank(G_lcc, weight="weight")
    deg = dict(G_lcc.degree())

    top_authors = pd.DataFrame({
        "author": list(G_lcc.nodes()),
        "pagerank": [pr[n] for n in G_lcc.nodes()],
        "degree": [deg[n] for n in G_lcc.nodes()],
    }).sort_values("pagerank", ascending=False).head(15)
    top_authors.to_csv("results/phase6_top_authors.csv", index=False)
    print("\nTop Authors by PageRank (Largest Connected Component):\n", top_authors.head(10).to_string(index=False))

# 2. Semantic Article Similarity Graph on Busiest Month
peak_month = df["month"].value_counts().idxmax()
sub = df[df["month"] == peak_month].head(1000).reset_index(drop=True)
Z_sub = Z[sub["id"].values]
sim_mat = cosine_similarity(Z_sub)

G_news = nx.Graph()
for i, row in sub.iterrows():
    G_news.add_node(i, category=row["category"])

rows_i, cols_j = np.where(np.triu(sim_mat, k=1) > 0.72)
for i, j in zip(rows_i, cols_j):
    G_news.add_edge(int(i), int(j), weight=float(sim_mat[i, j]))

communities = list(nx.community.greedy_modularity_communities(G_news))
same_cat_edges = sum(
    1 for u, v in G_news.edges() if G_news.nodes[u]["category"] == G_news.nodes[v]["category"]
)
homophily = same_cat_edges / max(G_news.number_of_edges(), 1)

summary = pd.DataFrame([{
    "peak_month": peak_month,
    "news_nodes": G_news.number_of_nodes(),
    "news_edges": G_news.number_of_edges(),
    "num_communities": len(communities),
    "category_homophily": round(homophily, 4),
    "avg_clustering": round(nx.average_clustering(G_news), 4),
}])
summary.to_csv("results/phase6_graph_summary.csv", index=False)
print("\nSemantic News Network Summary:\n", summary.to_string(index=False))

# Plot degree distribution of the semantic similarity graph
degrees = [d for _, d in G_news.degree()]
plt.figure(figsize=(7, 4))
plt.hist(degrees, bins=30, edgecolor="black")
plt.title(f"Degree Distribution of Semantic News Graph ({peak_month})")
plt.xlabel("Node Degree (Similar Articles > 0.72)")
plt.ylabel("Article Count")
plt.tight_layout()
plt.savefig("results/phase6_degree_dist.png")
plt.close()
print("Phase 6 complete! Saved graph tables and plot to results/.")
