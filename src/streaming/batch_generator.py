from collections.abc import Iterator
import pandas as pd

REQUIRED_COLUMNS = ["id", "date", "month", "category", "authors", "text"]

def create_monthly_batches(df: pd.DataFrame, min_batch_size: int = 1,)-> Iterator[tuple[str,pd.DataFrame]]:
    
    """
    Yield historical news articles one month at a time.

    Returns:
        (month_label, batch_dataframe)

    Example:
        ("2025-01", january_articles)
        ("2025-02", february_articles)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    
    missing = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )
        
    data=df.copy()
    
    data = data.sort_values(
        ["date", "id"]
    ).reset_index(drop=True)
    
    data["window"] = data["date"].dt.to_period("M").astype(str)
    
    for month, batch in data.groupby("month", sort=True):

        batch = batch.reset_index(drop=True)

        if len(batch) < min_batch_size:
            continue

        yield month, batch
        
def main():
    
    from data_loader import load_news
    
    df = load_news()
    batches = list(create_monthly_batches(df, min_batch_size=10))
    
    print("\nMONTHLY BATCH GENERATION SUCCESSFUL")
    print("-" * 45)

    print("Total articles:", len(df))
    print("Total monthly batches:", len(batches))

    for month, batch in batches:
        print(f"{month}: {len(batch)} articles")


if __name__ == "__main__":
    main()