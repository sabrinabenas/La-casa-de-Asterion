import pandas as pd


def normalize_transactions(df: pd.DataFrame):
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.astype({"STORE_ID": "category", "SKU": "category", "SUBGROUP": "category"})
    df = df.drop(columns="STORE_SUBGROUP_DATE_ID")
    return df


if __name__ == "__main__":
    from pathlib import Path

    files = list(Path("data").rglob("*.csv"))

    for file in files:
        pd.read_csv(file).to_parquet(file.with_suffix(".parquet"))
