
"""
End-to-end Streaming Data Intelligence Engine pipeline.

Run from project root:
    python -m src.pipeline

Execution order:
    1. Data cleaning
    2. Text representation
    3. Dimensionality reduction
    4. Topic modelling
    5. Graph analytics
    6. Temporal trend detection
    7. Event impact analysis
    8. Differential privacy
    9. Fairness and representation audit

The pipeline:
    - Runs each module in a separate Python process.
    - Uses the current virtual environment interpreter.
    - Stops when a stage fails.
    - Validates the actual output files of each module.
    - Supports skipping optional expensive stages.

Important:
    causal.py reads data/processed/events.csv directly.
    Create and populate that file before running the pipeline.

    The stream.py module simulates chronological batches from
    the static dataset. It is not a live ingestion service.
"""

import argparse
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"

EVENTS_PATH = DATA_DIR / "events.csv"


# ---------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------

STAGES = {
    "clean": {
        "command": ["-m", "src.clean"],
        "outputs": [
            DATA_DIR / "news.parquet",
        ],
        "description": "Clean and prepare the news dataset",
    },

    "represent": {
        "command": ["-m", "src.represent"],
        "outputs": [
            DATA_DIR / "emb.npy",
            DATA_DIR / "tfidf.npz",
            DATA_DIR / "tfidf_vectorizer.joblib",
        ],
        "description": "Generate embeddings and TF-IDF features",
    },

    "reduce": {
        "command": ["-m", "src.reduce"],
        "outputs": [
            DATA_DIR / "pca_embeddings.npy",
            DATA_DIR / "tsne_2d.npy",
        ],
        "description": "Perform dimensionality reduction",
    },

    "topics": {
        "command": ["-m", "src.topics"],
        "outputs": [
            RESULTS_DIR / "topics" / "article_topics.parquet",
            RESULTS_DIR / "topics" / "clustering_metrics.csv",
            RESULTS_DIR / "topics" / "topic_keywords.csv",
        ],
        "description": "Discover and evaluate article topics",
    },

    "graph": {
        "command": ["-m", "src.graph"],
        "outputs": [
            RESULTS_DIR / "graph" / "news_graph.graphml",
            RESULTS_DIR / "graph" / "article_graph_metrics.parquet",
        ],
        "description": "Build the article similarity graph",
    },

    "trends": {
        "command": ["-m", "src.trends"],
        "outputs": [
            RESULTS_DIR / "trends" / "category_shares.csv",
            RESULTS_DIR / "trends" / "category_drift.csv",
            RESULTS_DIR / "trends" / "semantic_drift.csv",
            RESULTS_DIR / "trends" / "topic_prevalence.csv",
        ],
        "description": "Detect temporal and semantic changes",
    },

    "causal": {
        "command": ["-m", "src.causal"],
        "outputs": [
            RESULTS_DIR / "causal" / "event_impact.csv",
            RESULTS_DIR / "causal" / "event_summary.csv",
        ],
        "description": "Analyze category changes around known events",
    },

    "privacy": {
        "command": ["-m", "src.privacy"],
        "outputs": [
            RESULTS_DIR / "privacy" / "private_category_statistics.csv",
            RESULTS_DIR / "privacy" / "privacy_metadata.json",
        ],
        "description": "Generate differentially private statistics",
    },

    "fairness": {
        "command": ["-m", "src.fairness"],
        "outputs": [
            RESULTS_DIR / "fairness" / "author_concentration.csv",
            RESULTS_DIR / "fairness" / "category_by_author_cohort.csv",
            RESULTS_DIR / "fairness" / "category_disparity.csv",
            RESULTS_DIR / "fairness" / "cohort_distribution_divergence.csv",
            RESULTS_DIR / "fairness" / "fairness_summary.json",
        ],
        "description": "Audit authorship and category representation",
    },
}


# ---------------------------------------------------------
# Output validation
# ---------------------------------------------------------

def validate_outputs(stage_name):
    """Check that all expected outputs exist and are non-empty."""

    expected_outputs = STAGES[stage_name]["outputs"]

    missing = []
    empty = []

    for path in expected_outputs:

        if not path.exists():
            missing.append(path)

        elif path.is_file() and path.stat().st_size == 0:
            empty.append(path)

    if missing or empty:

        print(
            f"\nERROR: Output validation failed for '{stage_name}'."
        )

        for path in missing:
            print(f"  Missing: {path}")

        for path in empty:
            print(f"  Empty: {path}")

        return False

    print(f"\nOutput validation passed: {stage_name}")

    for path in expected_outputs:
        print(f"  [OK] {path.relative_to(BASE_DIR)}")

    return True


# ---------------------------------------------------------
# Execute one stage
# ---------------------------------------------------------

def execute_stage(stage_name, extra_args=None):
    """Execute one module using the current Python interpreter."""

    stage = STAGES[stage_name]

    command = [
        sys.executable,
        *stage["command"],
        *(extra_args or []),
    ]

    print("\n" + "=" * 70)
    print(f"STAGE: {stage_name.upper()}")
    print(stage["description"])
    print("=" * 70)

    print("Command:")
    print(" ".join(f'"{part}"' if " " in part else part for part in command))
    print()

    try:
        subprocess.run(
            command,
            cwd=BASE_DIR,
            check=True,
        )

    except subprocess.CalledProcessError as error:
        print("\n" + "=" * 70)
        print(f"PIPELINE FAILED AT: {stage_name.upper()}")
        print(f"Exit code: {error.returncode}")
        print("=" * 70)

        return False

    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
        return False

    return validate_outputs(stage_name)


# ---------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------

def run_pipeline(args):
    """Execute selected pipeline stages in order."""

    skipped = set()

    if args.skip_graph:
        skipped.add("graph")

    if args.skip_causal:
        skipped.add("causal")

    if args.skip_privacy:
        skipped.add("privacy")

    if args.skip_fairness:
        skipped.add("fairness")

    # Validate the event file before starting expensive stages.
    if "causal" not in skipped and not EVENTS_PATH.exists():

        print("\nERROR: Event dataset not found.")
        print(f"Expected location: {EVENTS_PATH}")
        print(
            "\nRun causal.py once to create the template, "
            "then add documented events to events.csv."
        )

        return False

    print("\n" + "=" * 70)
    print("STREAMING DATA INTELLIGENCE ENGINE")
    print("END-TO-END PIPELINE")
    print("=" * 70)

    print(f"\nProject directory: {BASE_DIR}")
    print(f"Python interpreter: {sys.executable}")

    print("\nExecution plan:")

    for name, stage in STAGES.items():

        status = (
            "SKIPPED"
            if name in skipped
            else "PENDING"
        )

        print(
            f"  {name:<12} {status:<10} "
            f"- {stage['description']}"
        )

    completed = []

    for stage_name in STAGES:

        if stage_name in skipped:
            print(f"\nSkipping stage: {stage_name}")
            continue

        extra_args = []

        # Privacy configuration.
        if stage_name == "privacy":
            extra_args.extend([
                "--epsilon",
                str(args.epsilon),
            ])

            if args.privacy_random_state is not None:
                extra_args.extend([
                    "--random-state",
                    str(args.privacy_random_state),
                ])

        # No custom causal arguments are passed here.
        # The current causal.py reads its event file directly.

        success = execute_stage(
            stage_name,
            extra_args=extra_args,
        )

        if not success:

            print("\nPIPELINE STOPPED.")
            print(
                "Successfully completed stages:",
                completed
            )

            print(
                f"Failed stage: {stage_name}"
            )

            return False

        completed.append(stage_name)

    # -----------------------------------------------------
    # Final report
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("\nCompleted stages:")

    for stage in completed:
        print(f"  [OK] {stage}")

    if skipped:
        print("\nSkipped stages:")

        for stage in skipped:
            print(f"  [SKIPPED] {stage}")

    print("\nGenerated directories:")
    print(f"  Processed data: {DATA_DIR}")
    print(f"  Results:        {RESULTS_DIR}")

    print("\nAll selected pipeline stages completed.")

    return True


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Run the complete Streaming Data Intelligence "
            "Engine pipeline."
        )
    )

    parser.add_argument(
        "--skip-graph",
        action="store_true",
        help="Skip computationally expensive graph construction.",
    )

    parser.add_argument(
        "--skip-causal",
        action="store_true",
        help="Skip event impact analysis.",
    )

    parser.add_argument(
        "--skip-privacy",
        action="store_true",
        help="Skip differential privacy analysis.",
    )

    parser.add_argument(
        "--skip-fairness",
        action="store_true",
        help="Skip representation and fairness auditing.",
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0,
        help="Privacy budget per monthly histogram.",
    )

    parser.add_argument(
        "--privacy-random-state",
        type=int,
        default=None,
        help="Optional deterministic seed for privacy experiments.",
    )

    args = parser.parse_args()

    if args.epsilon <= 0:
        parser.error("--epsilon must be greater than zero.")

    success = run_pipeline(args)

    sys.exit(0 if success else 1)
