from pathlib import Path
from datetime import datetime,timezone
import os
import tempfile
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = BASE_DIR / "results" / "streaming" / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "engine_state.joblib"

def create_initial_state():
    """
    Create an initial state dictionary for the streaming engine.
    """
    return {
        "last_processed_month": None,
        "processed_months": [],
        "total_articles_processed": 0,

        # Topic tracking
        "topic_centroids": None,
        "topic_counts": {},
        "topic_names": {},
        "next_topic_id": 0,
        "initial_topic_count": 8,

        # Category trend tracking
        "previous_category_counts": None,

        # Semantic drift tracking
        "semantic_baseline_mean": None,
        "semantic_baseline_std": None,
        "previous_semantic_distribution": None,

        # Processing metadata
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    
class StreamingState:
    def __init__(self, checkpoint_path=CHECKPOINT_PATH):
        self.checkpoint_path = Path(checkpoint_path)
        self.state = create_initial_state()
        
    def save(self):
        """
        Save the current state safely using a temporary file
        and then replace the previous checkpoint.
        """

        self.checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.state["updated_at"] = (
            datetime.now(timezone.utc).isoformat()
        )

        temp_path = None

        try:
            with tempfile.NamedTemporaryFile(
                dir=self.checkpoint_path.parent,
                suffix=".tmp",
                delete=False,
            ) as temp_file:

                temp_path = Path(temp_file.name)

            joblib.dump(self.state, temp_path)

            os.replace(temp_path, self.checkpoint_path)

        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink()

    def load(self):
        """
        Restore state from an existing checkpoint.
        """

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"No checkpoint found: {self.checkpoint_path}"
            )

        loaded_state = joblib.load(self.checkpoint_path)

        if not isinstance(loaded_state, dict):
            raise ValueError("Invalid checkpoint format.")

        initial_state = create_initial_state()

        # Add missing keys without overwriting existing checkpoint data
        for key, value in initial_state.items():
            loaded_state.setdefault(key, value)

        self.state = loaded_state

        return self.state

    def update_after_batch(
        self,
        month,
        article_count,
        category_counts=None,
    ):
        """
        Update progress after successfully processing a batch.
        """

        if month in self.state["processed_months"]:
            raise ValueError(
                f"Month {month} has already been processed."
            )

        previous_month = self.state["last_processed_month"]

        if previous_month is not None and month <= previous_month:
            raise ValueError(
                "Batches must be processed chronologically."
            )

        self.state["last_processed_month"] = month

        self.state["processed_months"].append(month)

        self.state["total_articles_processed"] += article_count

        if category_counts is not None:
            self.state["previous_category_counts"] = category_counts

        self.save()

    def reset(self):
        """
        Reset the state to start a fresh experiment.
        """

        self.state = create_initial_state()

        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()

    def get_progress(self):
        """
        Return a readable summary of engine progress.
        """

        return {
            "last_processed_month": self.state["last_processed_month"],
            "processed_months": len(self.state["processed_months"]),
            "total_articles_processed": self.state[
                "total_articles_processed"
            ],
        }