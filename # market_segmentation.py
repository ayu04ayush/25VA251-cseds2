# market_segmentation.py
# Complete end-to-end script for Market Segmentation (K-Means and GMM)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings("ignore")

# ----------- USER: change this path to your CSV file ---------------
DATA_PATH = r"C:\Users\Asus\Downloads\online_retail_II.csv"
# -------------------------------------------------------------------

def load_data(path):
    df = pd.read_csv(path, encoding="latin1", low_memory=False)

    # Find date column names
    date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]

    if len(date_cols) > 0:
        try:
            df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
        except:
            df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], infer_datetime_format=True, errors="coerce")
    else:
        print("Warning: No invoice date/time column detected.")

    return df


def find_customer_col(df):
    candidates = [c for c in df.columns if "customer" in c.lower() or "cust" in c.lower()]

    if len(candidates) > 0:
        return candidates[0]

    # fallback numeric ID
    for c in df.columns:
        if df[c].dtype in [np.int64, np.float64] and df[c].nunique() > 5:
            if "price" not in c.lower() and "qty" not in c.lower():
                return c

    raise ValueError("Customer column not found!")


def preprocess(df):
    cust_col = find_customer_col(df)
    print(f"Customer Column Detected: {cust_col}")

    # Remove null customers
    df = df.dropna(subset=[cust_col])

    # Convert negative qty/price to positive if needed
    if "Quantity" in df.columns:
        df = df[df["Quantity"] > 0]

    if "UnitPrice" in df.columns:
        df = df[df["UnitPrice"] > 0]

    # Compute revenue
    if "Quantity" in df.columns and "UnitPrice" in df.columns:
        df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    # Group per customer
    grouped = df.groupby(cust_col).agg({
        "Revenue": "sum" if "Revenue" in df.columns else "mean",
        "Quantity": "sum" if "Quantity" in df.columns else "mean",
        "UnitPrice": "mean" if "UnitPrice" in df.columns else "mean"
    })

    grouped = grouped.fillna(0)

    return grouped


def scale_data(df):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    return scaled


def kmeans_clustering(data, k=4):
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(data)
    score = silhouette_score(data, labels)
    print(f"K-Means Silhouette Score: {score:.3f}")
    return labels


def gmm_clustering(data, k=4):
    model = GaussianMixture(n_components=k, random_state=42)
    labels = model.fit_predict(data)
    score = silhouette_score(data, labels)
    print(f"GMM Silhouette Score: {score:.3f}")
    return labels


def main():
    df = load_data(DATA_PATH)
    print("Data Loaded:", df.shape)

    processed = preprocess(df)
    print("Processed Shape:", processed.shape)

    scaled = scale_data(processed)

    kmeans_labels = kmeans_clustering(scaled, k=4)
    gmm_labels = gmm_clustering(scaled, k=4)

    processed["KMeans"] = kmeans_labels
    processed["GMM"] = gmm_labels

    print(processed.head())

    processed.to_csv("segmentation_output.csv", index=True)
    print("\nSaved segmentation_output.csv successfully!")


if __name__ == "__main__":
    main()
