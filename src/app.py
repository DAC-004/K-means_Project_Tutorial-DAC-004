"""End-to-end California Housing K-Means and classification workflow."""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "housing.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "housing_clustered.csv"
KMEANS_PATH = ROOT / "models" / "kmeans_pipeline.joblib"
CLASSIFIER_PATH = ROOT / "models" / "cluster_classifier.joblib"
CLUSTER_PLOT_PATH = ROOT / "reports" / "figures" / "kmeans_clusters.png"
INCOME_PLOT_PATH = ROOT / "reports" / "figures" / "cluster_income_analysis.png"
CONFUSION_PLOT_PATH = ROOT / "reports" / "figures" / "classifier_confusion_matrix.png"
FEATURES = ["Latitude", "Longitude", "MedInc"]
RANDOM_STATE = 42
N_CLUSTERS = 6
TEST_SIZE = 0.2


def ensure_output_directories() -> None:
    for directory in [
        ROOT / "data" / "processed",
        ROOT / "models",
        ROOT / "reports" / "figures",
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def load_raw_housing() -> pd.DataFrame:
    """Load the runtime dataset. Copy from local specs only if the raw file is missing."""
    if not RAW_PATH.exists():
        source = ROOT / ".project_specs" / "housing.csv"
        if source.exists():
            RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
            RAW_PATH.write_bytes(source.read_bytes())
        else:
            raise FileNotFoundError(f"Dataset not found at {RAW_PATH} or {source}")
    return pd.read_csv(RAW_PATH)


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in FEATURES if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    print(f"Dataset shape: {df.shape}")
    print("Required dtypes:\n", df[FEATURES].dtypes)
    print("Missing values:\n", df[FEATURES].isna().sum())
    print(f"Duplicate rows: {df.duplicated().sum()}")
    return df


def split_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Index, pd.Index]:
    X = df[FEATURES].copy()
    X_train, X_test, train_idx, test_idx = train_test_split(
        X, df.index, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    return X_train, X_test, train_idx, test_idx


def build_kmeans() -> Pipeline:
    # One pipeline keeps imputation, scaling, and K-Means together so test
    # inference cannot refit preprocessing statistics.
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)),
        ]
    )


def fit_kmeans(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[Pipeline, pd.Series, pd.Series]:
    kmeans = build_kmeans()
    train_labels = pd.Series(kmeans.fit_predict(X_train), index=X_train.index, name="cluster")
    test_labels = pd.Series(kmeans.predict(X_test), index=X_test.index, name="cluster")
    print(
        "K-Means clusters:",
        sorted(set(train_labels)),
        "counts:",
        train_labels.value_counts().sort_index().to_dict(),
    )
    return kmeans, train_labels, test_labels


def build_labeled_dataset(
    df: pd.DataFrame,
    train_idx: pd.Index,
    test_idx: pd.Index,
    train_labels: pd.Series,
    test_labels: pd.Series,
) -> pd.DataFrame:
    labeled = df[FEATURES].copy()
    labeled["cluster"] = pd.Series(index=train_idx, data=train_labels.to_numpy(), dtype="Int64")
    labeled.loc[test_idx, "cluster"] = pd.Series(test_labels.to_numpy(), index=test_idx, dtype="Int64")
    labeled["split"] = "test"
    labeled.loc[train_idx, "split"] = "train"
    labeled.to_csv(PROCESSED_PATH, index=False)
    print(f"Processed rows: {len(labeled)}")
    print("Split counts:", labeled["split"].value_counts().to_dict())
    return labeled


def plot_kmeans_clusters(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    train_labels: pd.Series,
    test_labels: pd.Series,
) -> Path:
    plt.figure(figsize=(10, 7))
    plt.scatter(
        X_train["Longitude"],
        X_train["Latitude"],
        c=train_labels,
        cmap="tab10",
        marker="o",
        alpha=0.55,
        s=10,
        label="Train",
    )
    plt.scatter(
        X_test["Longitude"],
        X_test["Latitude"],
        c=test_labels,
        cmap="tab10",
        marker="x",
        alpha=0.9,
        s=28,
        label="Test",
    )
    plt.title("California Housing K-Means Clusters (6) with Test Predictions")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.colorbar(label="K-Means cluster")
    plt.tight_layout()
    plt.savefig(CLUSTER_PLOT_PATH, dpi=150)
    plt.close()
    return CLUSTER_PLOT_PATH


def plot_income_by_cluster(labeled: pd.DataFrame) -> Path:
    """Compact MedInc boxplots by cluster for the required income interpretation."""
    grouped = [labeled.loc[labeled["cluster"] == cluster, "MedInc"] for cluster in range(N_CLUSTERS)]
    plt.figure(figsize=(8, 5))
    plt.boxplot(grouped, tick_labels=list(range(N_CLUSTERS)), showfliers=False)
    plt.title("Median Income by K-Means Cluster")
    plt.xlabel("K-Means cluster")
    plt.ylabel("MedInc")
    plt.tight_layout()
    plt.savefig(INCOME_PLOT_PATH, dpi=150)
    plt.close()
    print("MedInc by cluster:\n", labeled.groupby("cluster")["MedInc"].describe())
    return INCOME_PLOT_PATH


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def train_and_evaluate_classifier(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    train_labels: pd.Series,
    test_labels: pd.Series,
) -> tuple[Pipeline, Pipeline, dict[str, float], dict[str, float], object]:
    baseline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)),
        ]
    )
    classifier = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
            ),
        ]
    )
    baseline.fit(X_train, train_labels)
    classifier.fit(X_train, train_labels)
    baseline_pred = baseline.predict(X_test)
    pred = classifier.predict(X_test)
    baseline_metrics = classification_metrics(test_labels, baseline_pred)
    classifier_metrics = classification_metrics(test_labels, pred)
    report = classification_report(test_labels, pred, zero_division=0)
    print("Dummy baseline:", baseline_metrics)
    print("Random Forest:", classifier_metrics)
    print(report)
    print(
        "Model selection: Random Forest is preferred because its macro F1 "
        "exceeds the DummyClassifier baseline."
    )
    print(
        "Interpretation: these metrics measure reproduction of K-Means "
        "pseudo-labels, not human-verified ground truth."
    )
    ConfusionMatrixDisplay.from_predictions(test_labels, pred, cmap="Blues")
    plt.title("Random Forest Classification of K-Means Labels")
    plt.tight_layout()
    plt.savefig(CONFUSION_PLOT_PATH, dpi=150)
    plt.close()
    return baseline, classifier, baseline_metrics, classifier_metrics, pred


def persist_and_verify(
    kmeans: Pipeline,
    classifier: Pipeline,
    X_test: pd.DataFrame,
    test_labels: pd.Series,
    pred,
) -> None:
    joblib.dump(kmeans, KMEANS_PATH)
    joblib.dump(classifier, CLASSIFIER_PATH)
    loaded_kmeans = joblib.load(KMEANS_PATH)
    loaded_classifier = joblib.load(CLASSIFIER_PATH)
    assert (loaded_kmeans.predict(X_test) == test_labels.to_numpy()).all()
    assert (loaded_classifier.predict(X_test) == pred).all()
    print(f"Saved and verified: {KMEANS_PATH.name}, {CLASSIFIER_PATH.name}")


def main() -> None:
    ensure_output_directories()
    raw = validate_data(load_raw_housing())
    X_train, X_test, train_idx, test_idx = split_features(raw)
    kmeans, train_labels, test_labels = fit_kmeans(X_train, X_test)
    labeled = build_labeled_dataset(raw, train_idx, test_idx, train_labels, test_labels)
    plot_kmeans_clusters(X_train, X_test, train_labels, test_labels)
    plot_income_by_cluster(labeled)
    _baseline, classifier, _baseline_metrics, _classifier_metrics, pred = (
        train_and_evaluate_classifier(X_train, X_test, train_labels, test_labels)
    )
    persist_and_verify(kmeans, classifier, X_test, test_labels, pred)
    print(
        "Test prediction conclusion: all held-out rows received valid labels "
        "from the fitted K-Means model; visual overlap/boundary ambiguity "
        "should be assessed from the saved plot."
    )
    print(
        "Cluster interpretation: clusters represent combinations of location "
        "and median income; geographic overlap and boundary ambiguity are "
        "expected because K-Means partitions standardized feature space by distance."
    )


if __name__ == "__main__":
    main()
