from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from src.data_loader import load_news, load_embeddings
from src.streaming.batch_generator import create_monthly_batches
from src.streaming.state import StreamingState
from src.streaming.topic_tracker import TopicTracker
from src.streaming.topic_reporter import TopicReporter

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "results" / "streaming"/"batches"

def save_monthly_batch(month: str,batch,batch_embeddings):
    """Save one monthly batch and its corresponding embeddings."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    month_name = month.replace("/", "-")

    articles_path = OUTPUT_DIR / f"{month_name}_articles.parquet"
    embeddings_path = OUTPUT_DIR / f"{month_name}_embeddings.npy"
    summary_path = OUTPUT_DIR / f"{month_name}_summary.json"

    batch.to_parquet(articles_path, index=False)
    np.save(embeddings_path, batch_embeddings)

    summary = {
        "month": month,
        "article_count": int(len(batch)),
        "embedding_dimensions": int(batch_embeddings.shape[1]),
        "category_counts": {
            str(category): int(count)
            for category, count in batch["category"].value_counts().items()
        },
    }

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)


class StreamingEngine:
    def __init__(self):
        self.state = StreamingState()
    
    def run(self,resume:bool=True) -> None:
        """Run the streaming engine to process monthly batches of news articles."""
        checkpoint = Path(self.state.checkpoint_path)
        if resume and checkpoint.exists():
            self.state.load()
            print("Existing checkpoint loaded.")
        elif not resume:
            self.state.reset()
            print("Checkpoint reset.")
        else:
            print("No checkpoint found. Starting new stream.")

        df = load_news()
        embeddings = load_embeddings()
        
        tracker = TopicTracker(self.state)
        reporter = TopicReporter()
        
        batches=list(create_monthly_batches(df))
        print(f"Total monthly batches: {len(batches)}")
        
        processed_months = set(self.state.state["processed_months"])

        for month, batch in batches:
            if month in processed_months:
                print(f"Skipping already processed month: {month}")
                continue

            print(f"Processing month: {month} with {len(batch)} articles.")
            article_ids = batch["id"].to_numpy(dtype=int)
            if np.any(article_ids < 0) or np.any(article_ids >= len(embeddings)):
                raise ValueError(
                    f"Invalid article IDs found in batch {month}."
                )
            batch_embeddings = embeddings[article_ids]

             # Step 1: Assign articles to existing or emerging topics.
            topic_labels = tracker.process_batch(batch_embeddings)
            
            #step 2: Save articles and their embeddings
            save_monthly_batch(
                month=month,
                batch=batch,
                batch_embeddings=batch_embeddings,
            )
            
            # Step 3: Save topic assignments and monthly summary.
            reporter.save_monthly_report(
                month,
                batch,
                topic_labels,
                tracker,
            )
            # Step 4: Update checkpoint after outputs are saved.
            category_counts = {
                str(category): int(count)
                for category, count
                in batch["category"].value_counts().items()
            }
            
            self.state.update_after_batch(
                month=month,
                article_count=len(batch),
                category_counts=category_counts,
            )

            processed_months.add(month)
            
            emerging_count = int(
                np.sum(
                    topic_labels >= tracker.state.state[
                        "initial_topic_count"
                    ]
                )
            )

            unassigned_count = int(
                np.sum(topic_labels == -1)
            )
            print(f"Articles processed: {len(batch)}")
            print(f"Emerging-topic articles: {emerging_count}")
            print(f"Unassigned articles: {unassigned_count}")
            print("Monthly topic report saved.")
            print("Checkpoint updated.")

        print("\nStreaming completed.")
        print("Final progress:", self.state.get_progress())

        print("\nStreaming completed.")
        print("Final progress:", self.state.get_progress())

        print("\nCurrent topics:")
        for topic in tracker.get_topic_summary():
            print(topic)


def main():
    engine = StreamingEngine()
    engine.run(resume=True)


if __name__ == "__main__":
    main()