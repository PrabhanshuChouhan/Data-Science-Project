import numpy as np

from src.data_loader import load_news, load_embeddings

news = load_news()
embeddings = load_embeddings()

print("News shape:", news.shape)
print("Embedding shape:", embeddings.shape)

print("\nEmbedding statistics:")
print("Minimum:", np.min(embeddings))
print("Maximum:", np.max(embeddings))
print("Mean:", np.mean(embeddings))
print("Standard deviation:", np.std(embeddings))

norms = np.linalg.norm(embeddings, axis=1)

print("\nEmbedding norm statistics:")
print("Minimum:", np.min(norms))
print("Maximum:", np.max(norms))
print("Mean:", np.mean(norms))