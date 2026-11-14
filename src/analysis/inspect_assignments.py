import pandas as pd
from pathlib import Path

REPORT_DIR = Path("src/results/streaming/reports")

rows = []

for file in sorted(REPORT_DIR.glob("*_topics.parquet")):
    df = pd.read_parquet(file)

    rows.append({
        "month": file.stem.replace("_topics", ""),
        "articles": len(df),
        "unassigned_pct": round((df["topic_id"] == -1).mean() * 100, 1),
        "emerging": int((df["topic_type"] == "emerging").sum()),
        "existing": int((df["topic_type"] == "existing").sum()),
    })

result = pd.DataFrame(rows)

print(result.to_string(index=False))

print("\nOVERALL AVERAGES")
print(result[["unassigned_pct", "emerging", "existing"]].mean())