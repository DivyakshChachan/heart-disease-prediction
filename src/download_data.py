
from __future__ import annotations
import os
import sys
import ssl
import urllib.request
import pandas as pd

CANDIDATE_URLS = [
    # Common mirrors of Kaggle/processed UCI heart.csv (303 rows)
    "https://raw.githubusercontent.com/krishnaik06/Heart-Disease-Data-Analysis/master/heart.csv",
    "https://raw.githubusercontent.com/anikannal/heart_disease/master/heart.csv",
    "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/heart/heart.csv",
    "https://raw.githubusercontent.com/AVRahul/heart-disease-dataset/master/heart.csv",
    # Additional mirrors
    "https://raw.githubusercontent.com/datasets/heart-disease/master/data/heart.csv",
    "https://raw.githubusercontent.com/SR-Sunny-Raj/Hackerrank_solutions/main/Datasets/heart.csv",
]

EXPECTED_COLS = set(["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","target"]) 

def try_download(url: str, dest_path: str) -> bool:
    try:
        print(f"Attempting: {url}")
        # Use unverified SSL context to bypass local cert issues
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(url, timeout=30, context=context) as r, open(dest_path, 'wb') as f:
            f.write(r.read())
        # quick validation
        df = pd.read_csv(dest_path)
        if 'target' not in df.columns:
            # Some mirrors use 'num'; map to 'target'
            if 'num' in df.columns:
                df = df.rename(columns={'num': 'target'})
                df.to_csv(dest_path, index=False)
            else:
                raise ValueError("No target/num column present")
        missing = EXPECTED_COLS - set(df.columns)
        if missing:
            # Allow slight variations but require core set
            core = {"age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","target"}
            if not core.issubset(df.columns):
                raise ValueError(f"Missing core columns: {core - set(df.columns)}")
        print("Downloaded and validated dataset.")
        return True
    except Exception as e:
        print(f"Failed from {url}: {e}")
        try:
            os.remove(dest_path)
        except Exception:
            pass
        return False

def download_from_uci(dest_path: str) -> bool:
    """
    Download UCI processed Cleveland dataset and convert to expected columns.
    """
    try:
        print("Attempting UCI processed Cleveland dataset...")
        context = ssl._create_unverified_context()
        uci_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        with urllib.request.urlopen(uci_url, timeout=30, context=context) as r:
            raw = r.read().decode("utf-8")
        # Save raw temporarily
        tmp_path = dest_path + ".tmp"
        with open(tmp_path, "w") as f:
            f.write(raw)
        columns = [
            "age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","num"
        ]
        df = pd.read_csv(tmp_path, header=None, names=columns, na_values="?")
        # Binary target mapping: any disease (num > 0) => 1 else 0
        df["target"] = (df["num"].fillna(0) > 0).astype(int)
        df = df.drop(columns=["num"])  # drop original multi-class label
        # Reorder columns to expected schema
        df = df[[
            "age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","target"
        ]]
        df.to_csv(dest_path, index=False)
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        print("Downloaded and converted UCI Cleveland dataset.")
        return True
    except Exception as e:
        print(f"Failed from UCI Cleveland: {e}")
        try:
            os.remove(dest_path)
        except Exception:
            pass
        return False

def main():
    root = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(root, 'data')
    os.makedirs(data_dir, exist_ok=True)
    dest = os.path.join(data_dir, 'heart_disease_dataset.csv')
    for url in CANDIDATE_URLS:
        if try_download(url, dest):
            print(f"Saved to {dest}")
            return 0
    # Try UCI Cleveland fallback
    if download_from_uci(dest):
        print(f"Saved to {dest}")
        return 0
    print("All downloads failed. Please manually place 'heart_disease_dataset.csv' under data/ containing the UCI/Kaggle heart.csv columns.")
    return 1

if __name__ == "__main__":
    sys.exit(main())
