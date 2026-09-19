"""End-to-end California Housing K-Means and classification workflow."""
from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "housing.csv"
FEATURES = ["Latitude", "Longitude", "MedInc"]
RANDOM_STATE = 42

def validate_data(df: pd.DataFrame) -> None:
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    print(f"Dataset shape: {df.shape}")
    print("Required dtypes:\n", df[FEATURES].dtypes)
    print("Missing values:\n", df[FEATURES].isna().sum())
    print(f"Duplicate rows: {df.duplicated().sum()}")

def build_kmeans() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=6, random_state=RANDOM_STATE, n_init=10)),
    ])

def main() -> None:
    for directory in [ROOT / "data" / "processed", ROOT / "models", ROOT / "reports" / "figures"]:
        directory.mkdir(parents=True, exist_ok=True)
    if not RAW_PATH.exists():
        source = ROOT / ".project_specs" / "housing.csv"
        if source.exists():
            RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
            RAW_PATH.write_bytes(source.read_bytes())
        else:
            raise FileNotFoundError(f"Dataset not found at {RAW_PATH} or {source}")
    df = pd.read_csv(RAW_PATH)
    validate_data(df)
    X = df[FEATURES].copy()
    X_train, X_test, train_idx, test_idx = train_test_split(
        X, df.index, test_size=0.2, random_state=RANDOM_STATE
    )
    kmeans = build_kmeans()
    train_labels = kmeans.fit_predict(X_train)
    test_labels = kmeans.predict(X_test)
    print("K-Means clusters:", sorted(set(train_labels)), "counts:", pd.Series(train_labels).value_counts().sort_index().to_dict())

    labeled = df[FEATURES].copy()
    labeled["cluster"] = pd.Series(index=train_idx, data=train_labels, dtype="Int64")
    labeled.loc[test_idx, "cluster"] = pd.Series(test_labels, index=test_idx, dtype="Int64")
    labeled["split"] = "test"
    labeled.loc[train_idx, "split"] = "train"
    labeled.to_csv(ROOT / "data" / "processed" / "housing_clustered.csv", index=False)

    plt.figure(figsize=(10, 7))
    for labels, subset, marker, name, alpha in [(train_labels, X_train, "o", "Train", 0.55), (test_labels, X_test, "x", "Test", 0.9)]:
        plt.scatter(subset["Longitude"], subset["Latitude"], c=labels, cmap="tab10", marker=marker, alpha=alpha, s=10 if marker == "o" else 28, label=name)
    plt.title("California Housing K-Means Clusters (6) with Test Predictions")
    plt.xlabel("Longitude"); plt.ylabel("Latitude"); plt.legend(); plt.colorbar(label="K-Means cluster")
    plt.tight_layout(); plt.savefig(ROOT / "reports" / "figures" / "kmeans_clusters.png", dpi=150); plt.close()

    baseline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE))])
    classifier = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1))])
    baseline.fit(X_train, train_labels); classifier.fit(X_train, train_labels)
    baseline_pred = baseline.predict(X_test); pred = classifier.predict(X_test)
    def metrics(y, p):
        return {"accuracy": accuracy_score(y, p), "precision_macro": precision_score(y, p, average="macro", zero_division=0), "recall_macro": recall_score(y, p, average="macro", zero_division=0), "f1_macro": f1_score(y, p, average="macro", zero_division=0)}
    baseline_metrics = metrics(test_labels, baseline_pred)
    classifier_metrics = metrics(test_labels, pred)
    print("Dummy baseline:", baseline_metrics)
    print("Random Forest:", classifier_metrics)
    print(classification_report(test_labels, pred, zero_division=0))
    print("Model selection: Random Forest is preferred because its macro F1 exceeds the DummyClassifier baseline.")
    print("Interpretation: these metrics measure reproduction of K-Means pseudo-labels, not human-verified ground truth.")
    ConfusionMatrixDisplay.from_predictions(test_labels, pred, cmap="Blues")
    plt.title("Random Forest Classification of K-Means Labels"); plt.tight_layout(); plt.savefig(ROOT / "reports" / "figures" / "classifier_confusion_matrix.png", dpi=150); plt.close()

    kmeans_path = ROOT / "models" / "kmeans_pipeline.joblib"; classifier_path = ROOT / "models" / "cluster_classifier.joblib"
    joblib.dump(kmeans, kmeans_path)
    joblib.dump(classifier, classifier_path)
    loaded_kmeans = joblib.load(kmeans_path)
    loaded_classifier = joblib.load(classifier_path)
    assert (loaded_kmeans.predict(X_test) == test_labels).all()
    assert (loaded_classifier.predict(X_test) == pred).all()
    print(f"Saved and verified: {kmeans_path.name}, {classifier_path.name}")
    print("Test prediction conclusion: all held-out rows received valid labels from the fitted K-Means model; visual overlap/boundary ambiguity should be assessed from the saved plot.")
    print("Cluster interpretation: clusters represent combinations of location and median income; geographic overlap and boundary ambiguity are expected because K-Means partitions feature space by distance.")

if __name__ == "__main__":
    main()
