
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer


BASE_DIR = Path.cwd()

REPORT_DIR = BASE_DIR / "src" / "results" / "streaming" / "reports"
BATCH_DIR = BASE_DIR / "src" / "results" / "streaming" / "batches"

OUTPUT_DIR = BASE_DIR / "results" / "streaming" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 1. Load monthly topic reports
# --------------------------------------------------

def load_reports():

    files = sorted(REPORT_DIR.glob("*_topics.parquet"))

    if not files:
        raise FileNotFoundError(
            f"No monthly topic reports found in {REPORT_DIR}"
        )

    reports = []

    for file in files:

        df = pd.read_parquet(file)

        df["month"] = file.stem.replace("_topics", "")

        reports.append(df)

    data = pd.concat(reports, ignore_index=True)

    data["topic_id"] = data["topic_id"].astype(int)

    print(f"Loaded {len(data):,} articles.")
    print(f"Loaded {len(files)} monthly reports.")

    return data


def calculate_topic_names(data):
    """Build readable topic labels from dominant categories and headline terms."""

    assigned = data[data["topic_id"] != -1].copy()

    if assigned.empty:
        return {}

    assigned["headline"] = (
        assigned["text"]
        .fillna("")
        .astype(str)
        .str.split(r"\.\s+", n=1)
        .str[0]
    )

    topic_documents = (
        assigned.groupby("topic_id")["headline"]
        .agg(" ".join)
        .sort_index()
    )

    boilerplate = {
        "photo", "photos", "picture", "pictures", "images", "look", "looks",
        "day", "days", "tip", "tips", "time", "rise", "huffpost",
        "huffington", "post", "story", "stories", "read", "watch", "video",
        "news", "newsbrief", "says", "said", "told", "new", "year", "years",
        "make", "ways", "best", "like", "daily", "email", "roundup",
        "morning", "headline", "today", "week", "weeks", "things", "thing",
        "want", "going", "comes", "coming", "think", "help", "say", "saying",
        "get", "got", "take", "takes", "just", "does", "did", "really", "way",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
        "sunday", "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "sing", "sings", "song", "songs", "know", "need", "quiz", "jan",
        "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct",
        "nov", "dec", "success", "death", "dead", "died", "brother",
        "action", "tweet", "tweets", "hill", "huffpollster",
    }

    vectorizer = TfidfVectorizer(
        stop_words=sorted(set(ENGLISH_STOP_WORDS) | boilerplate),
        ngram_range=(1, 2),
        max_features=30_000,
        sublinear_tf=True,
    )
    topic_terms = vectorizer.fit_transform(topic_documents)
    feature_names = vectorizer.get_feature_names_out()

    category_counts = (
        assigned.groupby(["topic_id", "category"])
        .size()
        .rename("article_count")
        .reset_index()
        .sort_values(
            ["topic_id", "article_count", "category"],
            ascending=[True, False, True],
        )
        .drop_duplicates("topic_id")
        .set_index("topic_id")["category"]
        .to_dict()
    )

    acronyms = {
        "Dsm": "DSM",
        "Epa": "EPA",
        "Fbi": "FBI",
        "Gop": "GOP",
        "Isis": "ISIS",
        "Lgbtq": "LGBTQ",
        "Nato": "NATO",
        "Nra": "NRA",
        "Us": "US",
    }

    curated_names = {
        3: "Food & Drink: Recipes and Cooking",
        5: "Wellness: Memoirs and Personal Essays",
        16: "Entertainment: Christmas Movies and Holiday Stories",
        23: "Politics: Daily News Briefings",
    }

    topic_names = {}
    used_names = set()

    for row_index, topic_id in enumerate(topic_documents.index):
        category = str(category_counts.get(topic_id, "Topic")).title()
        category_tokens = set(category.lower().replace("&", " ").split())
        scores = topic_terms.getrow(row_index).toarray().ravel()
        selected_terms = []
        term_order = sorted(
            range(len(scores)),
            key=lambda index: (
                scores[index]
                * (1.1 if " " in feature_names[index] else 1.0),
                scores[index],
            ),
            reverse=True,
        )

        for term_index in term_order:
            if scores[term_index] <= 0:
                break

            term = str(feature_names[term_index])

            if any(character.isdigit() for character in term):
                continue

            term_tokens = set(term.split())
            if term in category_tokens:
                continue

            if any(
                term_tokens & set(selected.split())
                for selected in selected_terms
            ):
                continue

            selected_terms.append(term)
            if len(selected_terms) == 3:
                break

        formatted_terms = [
            " ".join(
                acronyms.get(word, word)
                for word in term.title().split()
            )
            for term in selected_terms
        ]

        if formatted_terms:
            topic_name = f"{category}: {', '.join(formatted_terms[:2])}"
            if topic_name in used_names and len(formatted_terms) > 2:
                topic_name = f"{category}: {', '.join(formatted_terms)}"
        else:
            topic_name = f"{category} Topic"

        topic_name = curated_names.get(int(topic_id), topic_name)

        if topic_name in used_names:
            topic_name = f"{topic_name} ({int(topic_id)})"

        topic_names[int(topic_id)] = topic_name
        used_names.add(topic_name)

    return topic_names


# --------------------------------------------------
# 2. Topic activity across months
# --------------------------------------------------

def calculate_topic_activity(data, topic_names=None):

    activity = (
        data[data["topic_id"] != -1]
        .groupby(["month", "topic_id"])
        .size()
        .reset_index(name="article_count")
    )

    if topic_names is not None:
        activity["topic_name"] = activity["topic_id"].map(topic_names)

    return activity


# --------------------------------------------------
# 3. Topic stability
# --------------------------------------------------

def calculate_topic_stability(activity):

    stability_records = []

    months = sorted(activity["month"].unique())

    for i in range(1, len(months)):

        previous_month = months[i - 1]
        current_month = months[i]

        previous = activity[
            activity["month"] == previous_month
        ]

        current = activity[
            activity["month"] == current_month
        ]

        previous_topics = set(previous["topic_id"])
        current_topics = set(current["topic_id"])

        union = previous_topics | current_topics
        intersection = previous_topics & current_topics

        jaccard = (
            len(intersection) / len(union)
            if union else 1.0
        )

        stability_records.append({
            "previous_month": previous_month,
            "current_month": current_month,
            "shared_topics": len(intersection),
            "total_unique_topics": len(union),
            "topic_jaccard_stability": jaccard
        })

    return pd.DataFrame(stability_records)


# --------------------------------------------------
# 4. Emerging topic statistics
# --------------------------------------------------

def calculate_emerging_statistics(data):

    emerging = data[data["topic_type"] == "emerging"]

    monthly = (
        emerging.groupby("month")
        .agg(
            emerging_articles=("id", "count"),
            emerging_topics=("topic_id", "nunique")
        )
        .reset_index()
    )

    total_articles = len(data)

    summary = {
        "total_emerging_articles": int(len(emerging)),
        "total_emerging_topics": int(emerging["topic_id"].nunique()),
        "emerging_article_percentage": (
            round(len(emerging) / total_articles * 100, 2)
            if total_articles else 0
        )
    }

    return monthly, summary


# --------------------------------------------------
# 5. Topic growth detection
# --------------------------------------------------

def calculate_topic_growth(activity):

    pivot = activity.pivot_table(
        index="month",
        columns="topic_id",
        values="article_count",
        fill_value=0
    ).sort_index()

    growth_records = []
    topic_names = {}

    if "topic_name" in activity.columns:
        topic_names = (
            activity.drop_duplicates("topic_id")
            .set_index("topic_id")["topic_name"]
            .to_dict()
        )

    for topic_id in pivot.columns:

        counts = pivot[topic_id]

        for i in range(1, len(counts)):

            previous = int(counts.iloc[i - 1])
            current = int(counts.iloc[i])

            growth_percentage = (
                ((current - previous) / previous) * 100
                if previous > 0 else None
            )

            growth_records.append({
                "month": counts.index[i],
                "topic_id": int(topic_id),
                "topic_name": topic_names.get(
                    int(topic_id),
                    f"Topic {topic_id}",
                ),
                "previous_articles": previous,
                "current_articles": current,
                "growth_percentage": growth_percentage
            })

    return pd.DataFrame(growth_records)


# --------------------------------------------------
# 6. Processing performance
# --------------------------------------------------

def calculate_processing_performance():

    files = sorted(BATCH_DIR.glob("*_summary.json"))

    records = []

    for file in files:

        with open(file, "r", encoding="utf-8") as f:
            summary = json.load(f)

        article_count = int(
            summary.get("article_count", 0)
        )

        processing_seconds = summary.get(
            "processing_seconds"
        )

        records.append({
            "month": file.stem.replace("_summary", ""),
            "article_count": article_count,
            "processing_seconds": processing_seconds,
            "articles_per_second": (
                article_count / processing_seconds
                if processing_seconds and processing_seconds > 0
                else None
            )
        })

    return pd.DataFrame(records)


# --------------------------------------------------
# 7. Main evaluation
# --------------------------------------------------

def run_evaluation():

    start_time = time.perf_counter()

    print("Loading reports...")

    data = load_reports()

    print("Calculating topic activity...")

    topic_names = calculate_topic_names(data)
    activity = calculate_topic_activity(data, topic_names)

    print("Calculating topic stability...")

    stability = calculate_topic_stability(activity)

    print("Calculating emerging topic statistics...")

    emerging_monthly, emerging_summary = (
        calculate_emerging_statistics(data)
    )

    print("Calculating topic growth...")

    growth = calculate_topic_growth(activity)

    print("Calculating processing performance...")

    performance = calculate_processing_performance()

    # Save detailed results.

    activity.to_csv(
        OUTPUT_DIR / "topic_activity.csv",
        index=False
    )

    stability.to_csv(
        OUTPUT_DIR / "topic_stability.csv",
        index=False
    )

    emerging_monthly.to_csv(
        OUTPUT_DIR / "emerging_topics.csv",
        index=False
    )

    growth.to_csv(
        OUTPUT_DIR / "topic_growth.csv",
        index=False
    )

    performance.to_csv(
        OUTPUT_DIR / "processing_performance.csv",
        index=False
    )

    total_time = time.perf_counter() - start_time

    summary = {
        "total_articles": int(len(data)),
        "total_months": int(data["month"].nunique()),
        "total_topics": int(data.loc[
            data["topic_id"] != -1, "topic_id"
        ].nunique()),
        "unassigned_articles": int(
            (data["topic_id"] == -1).sum()
        ),
        "average_topic_stability": (
            round(
                float(stability["topic_jaccard_stability"].mean()),
                4
            )
            if not stability.empty else None
        ),
        **emerging_summary,
        "evaluation_runtime_seconds": round(total_time, 2)
    }

    with open(
        OUTPUT_DIR / "evaluation_summary.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(summary, f, indent=4)

    print("\nEvaluation completed.")
    print(json.dumps(summary, indent=4))

    print(f"\nResults saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_evaluation()
