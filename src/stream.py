def stream(df):
    for month, g in df.groupby('month'):
        yield month, g