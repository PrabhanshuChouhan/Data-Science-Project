import json
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="News Stream Topic & Trend Engine", layout="wide")
st.title("News Stream Topic Discovery & Statistical Trend Detection")

tab1, tab2, tab3, tab4 = st.tabs([
    "Trend Detection (Phase 7)",
    "EM Topics & Representation (Phases 3–5)",
    "Semantic t-SNE Map (Phase 4)",
    "Network Analysis (Phase 6)"
])

with tab1:
    st.subheader("Benchmark vs. Phase 2 Injected Ground-Truth Spikes")
    bench = pd.read_csv("results/phase7_benchmark_metrics.csv")
    st.dataframe(bench, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top Category Surges (BH q < 0.05)")
        cat_trends = pd.read_csv("results/phase7_category_trends.csv")
        st.dataframe(cat_trends[cat_trends["q_bh"] < 0.05].sort_values("bf10", ascending=False).head(15), use_container_width=True)
    with col2:
        st.subheader("Top Unsupervised EM Topic Surges (BH q < 0.05)")
        top_trends = pd.read_csv("results/phase7_topic_trends.csv")
        st.dataframe(top_trends[top_trends["q_bh"] < 0.05].sort_values("bf10", ascending=False).head(15), use_container_width=True)

    st.image("results/phase7_detected_spikes.png", caption="Detected Category Spikes (BH q < 0.05)")

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Phase 3: Embedding vs. Lexical Retrieval (kNN@10)")
        rep = pd.read_csv("results/phase3_representation_results.csv")
        st.dataframe(rep, use_container_width=True)
    with c2:
        st.subheader("Phase 5: From-Scratch DiagGMM vs. Baselines")
        nmi = pd.read_csv("results/phase5_em_vs_kmeans.csv")
        st.dataframe(nmi, use_container_width=True)

    st.subheader("Monthly Topic Share Evolution")
    shares = pd.read_csv("results/topic_shares.csv")
    fig_shares = px.line(shares, x="month", y="share", color="topic", title="Discovered Topic Shares Over Time")
    st.plotly_chart(fig_shares, use_container_width=True)

with tab3:
    st.subheader("2D t-SNE Projection of PCA-50 Article Embeddings")
    tsne_df = pd.read_csv("results/tsne_sample.csv")
    fig_tsne = px.scatter(tsne_df, x="x", y="y", color="category", opacity=0.7, height=600)
    st.plotly_chart(fig_tsne, use_container_width=True)

with tab4:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Top Co-Author Nodes by PageRank")
        authors = pd.read_csv("results/phase6_top_authors.csv")
        st.dataframe(authors, use_container_width=True)
    with c2:
        st.subheader("Semantic Similarity Network Summary")
        g_sum = pd.read_csv("results/phase6_graph_summary.csv")
        st.dataframe(g_sum, use_container_width=True)
        st.image("results/phase6_degree_dist.png")
