import pandas as pd

def load_tsv(path: str, head_rows=5, top_n=5)-> pd.DataFrame:
    print(f"📄 File: {path}")
    print("-" * 60)

    # Load TSV
    df = pd.read_csv(path, sep="\t", engine="python", on_bad_lines="warn")

    # Basic shape
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")

    # Column types
    print("\n📌 Data types:")
    print(df.dtypes)

    # Missing values
    missing = df.isna().sum()
    if missing.any():
        print("\n⚠️ Missing values:")
        print(missing[missing > 0])
    else:
        print("\nNo missing values.")

    # Memory usage
    print("\n🧠 Memory usage:")
    print(df.memory_usage(deep=True).sum() / (1024 ** 2), "mega_bytes")


    # Head preview
    print("\n👀 Head:")
    print(df.head())

    return df

def save_tsv(df: pd.DataFrame, path: str):
    df.to_csv(path, sep="\t", index=False)
    print(f"✅ DataFrame saved to {path}")  