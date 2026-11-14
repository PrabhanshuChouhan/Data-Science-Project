from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN,KMeans
from sklearn.preprocessing import normalize

class TopicTracker:
    def __init__(
        self,
        state,
        initial_topics: int = 8,
        similarity_threshold: float = 0.35,
        emerging_min_samples: int = 5,
        emerging_eps=0.40,
        refinement_threshold=30,
        refinement_topics=5
    ):
        self.state = state
        self.initial_topics = initial_topics
        self.similarity_threshold = similarity_threshold
        self.emerging_min_samples = emerging_min_samples
        self.emerging_eps = emerging_eps
        self.emerging_topic_match_threshold = 0.45
        self.refinement_threshold = refinement_threshold
        self.refinement_topics = refinement_topics
        
        self.state.state.setdefault("topic_counts", {})
        self.state.state.setdefault("topic_names", {})
        self.state.state.setdefault("initial_topic_count",initial_topics)
    
    def _get_centroids(self):
        centroids = self.state.state["topic_centroids"]

        if centroids is None:
            return None

        return normalize(np.asarray(centroids, dtype=float))
    
    def _initialize_topics(self, embeddings: np.ndarray) -> np.ndarray:
        """Discover the initial topics from the first monthly batch."""
        
        number_of_topics = min(self.initial_topics, len(embeddings))

        model = KMeans(n_clusters=number_of_topics, random_state=42,n_init=18)
        labels = model.fit_predict(embeddings)
        centroids = normalize(model.cluster_centers_)

        self.state.state["topic_centroids"] = centroids.tolist()
        self.state.state["topic_counts"] = {
            int(topic_id): int(np.sum(labels == topic_id))
            for topic_id in range(number_of_topics)
        }
        self.state.state["topic_names"] = {
            int(topic_id): f"Topic {topic_id}"
            for topic_id in range(number_of_topics)
        }

        self.state.state["next_topic_id"] = number_of_topics
        self.state.state["initial_topic_count"] = number_of_topics
        
        return labels.astype(int)
    
    def _update_centroid(self,topic_id:int, new_embeddings:np.ndarray)->None:
        """Update the centroid of a topic with a new embedding."""
        centroids = np.asarray(
            self.state.state["topic_centroids"],
            dtype=float,
        )
        counts = self.state.state["topic_counts"]

        old_count = int(counts.get(topic_id, 0))
        new_count = len(new_embeddings)

        old_centroid = centroids[topic_id]

        new_mean = new_embeddings.mean(axis=0)

        updated_centroid = (
            old_centroid * old_count + new_mean * new_count
        ) / (old_count + new_count)

        updated_centroid = normalize(
            updated_centroid.reshape(1, -1)
        )[0]

        centroids[topic_id] = updated_centroid

        self.state.state["topic_centroids"] = centroids
        counts[topic_id] = old_count + new_count

    def _create_emerging_topics(self, embeddings, article_labels):

        if len(embeddings) < self.emerging_min_samples:
            return article_labels

        clustering = DBSCAN(
            eps=self.emerging_eps,
            min_samples=self.emerging_min_samples,
            metric="cosine"
        )

        cluster_labels = clustering.fit_predict(embeddings)

        for cluster_id in sorted(set(cluster_labels)):

            if cluster_id == -1:
                continue

            positions = np.where(cluster_labels == cluster_id)[0]
            cluster_embeddings = embeddings[positions]

            if len(positions) >= self.refinement_threshold:

                number_of_subtopics = min(
                    self.refinement_topics,
                    max(2, len(positions) // self.refinement_threshold)
                )

                refinement = KMeans(
                    n_clusters=number_of_subtopics,
                    random_state=42,
                    n_init=10
                )

                subtopic_labels = refinement.fit_predict(
                    cluster_embeddings
                )

            else:
                subtopic_labels = np.zeros(len(positions), dtype=int)

            for subtopic_id in sorted(set(subtopic_labels)):

                sub_positions = positions[
                    subtopic_labels == subtopic_id
                ]

                sub_embeddings = embeddings[sub_positions]

                new_centroid = normalize(
                    sub_embeddings.mean(axis=0).reshape(1, -1)
                )[0]

                centroids = self._get_centroids()

                similarities = np.dot(
                    centroids,
                    new_centroid
                )

                best_position = int(np.argmax(similarities))
                best_similarity = float(similarities[best_position])

                # Reuse an existing topic if sufficiently similar
                if best_similarity >= self.emerging_topic_match_threshold:

                    topic_id = best_position

                    self._update_centroid(
                        topic_id,
                        sub_embeddings
                    )

                else:

                    topic_id = int(
                        self.state.state["next_topic_id"]
                    )

                    centroids = np.asarray(
                        self.state.state["topic_centroids"],
                        dtype=float
                    )

                    self.state.state["topic_centroids"] = np.vstack(
                        [centroids, new_centroid]
                    )

                    self.state.state["topic_counts"][topic_id] = 0

                    self.state.state["topic_names"][topic_id] = (
                        f"Emerging Topic {topic_id}"
                    )

                    self.state.state["next_topic_id"] = topic_id + 1

                    self._update_centroid(
                        topic_id,
                        sub_embeddings
                    )

                article_labels[sub_positions] = topic_id

        return article_labels
    def process_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """Process a batch of embeddings and assign topic labels."""
        embeddings = np.asarray(embeddings, dtype=float)
        embeddings = normalize(embeddings)
        centroids = self._get_centroids()
        if centroids is None:
            return self._initialize_topics(embeddings)

        if embeddings.shape[1] != centroids.shape[1]:
            raise ValueError(
                "Embedding dimensions do not match existing topic centroids."
            )
        
        similarities = np.dot(embeddings, centroids.T)
        best_topic_ids = np.argmax(similarities, axis=1)
        best_similarities = np.max(similarities, axis=1)

        article_labels = np.full(len(embeddings), -1, dtype=int)
        
        familiar_mask = (
            best_similarities >= self.similarity_threshold
        )

        article_labels[familiar_mask] = best_topic_ids[familiar_mask]

        for topic_id in np.unique(article_labels[familiar_mask]):
            positions = np.where(article_labels == topic_id)[0]

            self._update_centroid(
                int(topic_id),
                embeddings[positions],
            )
            
        unfamiliar_positions = np.where(~familiar_mask)[0]

        if len(unfamiliar_positions) > 0:

            unfamiliar_embeddings = embeddings[unfamiliar_positions]

            unfamiliar_labels = np.full(
                len(unfamiliar_positions),
                -1,
                dtype=int
            )

            self._create_emerging_topics(
                unfamiliar_embeddings,
                unfamiliar_labels
            )

            article_labels[unfamiliar_positions] = unfamiliar_labels

        return article_labels
    
    def get_topic_summary(self) -> list[dict]:
        """Return the current topic information."""

        counts = self.state.state["topic_counts"]
        names = self.state.state["topic_names"]

        return [
            {
                "topic_id": int(topic_id),
                "topic_name": names.get(
                    topic_id,
                    f"Topic {topic_id}",
                ),
                "article_count": int(count),
            }
            for topic_id, count in sorted(counts.items())
        ]