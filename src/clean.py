import pandas as pd, numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "News_Category_Dataset_v3.json"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "news.parquet"


df = pd.read_json(DATA_PATH, lines=True)
df["date"] = pd.to_datetime(df["date"])
# keep the years with enough volume
df = df[(df.date >= "2012-01-01") & (df.date < "2019-01-01")]
# merge near-identical categories
df["category"] = df["category"].replace({
    "HEALTHY LIVING": "WELLNESS",
    "THE WORLDPOST": "WORLDPOST"
})
# build text, clean, drop junk and exact duplicates
df["text"] = (df.headline.fillna("") + ". " + df.short_description.fillna(""))
df["text"] = (df.text.str.lower()
              .str.replace(r"http\S+", "", regex=True)
              .str.strip())
df = df[df.text.str.len() > 20].drop_duplicates("text")

# keep the 8 biggest categories
top = df.category.value_counts().head(8).index
df = df[df.category.isin(top)]
df = df.sort_values("date").reset_index(drop=True)
df["id"] = np.arange(len(df))  # row id, used to index embeddings
df["month"] = df.date.dt.to_period("M").astype(str)

df[["id", "date", "month", "category", "authors", "text"]].to_parquet(OUTPUT_PATH,
    index=False)

print(df.shape, df.month.nunique())
print("Min items per month:", df.groupby("month").size().min())
