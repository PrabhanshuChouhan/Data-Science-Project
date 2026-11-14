from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = (BASE_DIR / "results" / "streaming" / "reports")

class TopicReporter:
    
    def __init__(self):
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        
    def save_monthly_report(self, month: str, batch: pd.DataFrame, topic_labels: np.ndarray,topic_tracker,):
        
        """Save the monthly report as a JSON file."""
        if len(batch) != len(topic_labels):
            raise ValueError(
                "Number of topic labels does not match articles."
            )
        required = ["id", "date", "month", "category", "text"]
        report = batch[required].copy()

        # Add topic assignments before using the topic_id column.
        report["topic_id"] = topic_labels.astype(int)
        initial_count = topic_tracker.state.state.get(
            "initial_topic_count",
            topic_tracker.initial_topics,
        )
        
        report["topic_type"] = np.where(
            report["topic_id"] == -1,
            "unassigned",
            np.where(
                report["topic_id"] >= initial_count,
                "emerging",
                "existing",
            ),
        )
    
        report_path = REPORT_DIR / f"{month}_topics.parquet"

        report.to_parquet(report_path, index=False)

        summary = (
            report.groupby(
                ["topic_id", "topic_type"],
                as_index=False
            )
            .agg(
                article_count=("id", "count"),
                categories=("category", lambda values: sorted(
                    values.dropna().astype(str).unique().tolist()
                )),
            )
        )

        summary_path = REPORT_DIR / f"{month}_topic_summary.json"

        summary.to_json(
            summary_path,
            orient="records",
            indent=4
        )

        return report, summary
