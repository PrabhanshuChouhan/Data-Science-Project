import subprocess
import sys

STEPS = [
    ("Phase 3: Embeddings & Distance Comparison", [sys.executable, "-m", "src.represent"]),
    ("Phase 4: PCA/SVD, LSH vs FAISS & t-SNE", [sys.executable, "-m", "src.reduce"]),
    ("Phase 5: DiagGMM EM Topic Modeling", [sys.executable, "-m", "src.topics"]),
    ("Phase 6: Co-Authorship & Semantic Graphs", [sys.executable, "-m", "src.graph"]),
    ("Phase 7: Statistical Trend Detection", [sys.executable, "-m", "src.trends"]),
]

if __name__ == "__main__":
    for title, cmd in STEPS:
        print(f"\n=== Running {title} ===")
        subprocess.run(cmd, check=True)
    print("\nAll pipeline phases completed successfully! Outputs saved in results/.")
