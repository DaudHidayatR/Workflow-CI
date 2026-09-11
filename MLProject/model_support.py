"""Shared data contract and evaluation semantics; no model selection here."""

import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import mlflow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "sqli_xss_preprocessing"


def configure():
    """Configure MLflow tracking and the experiment used by this training stage."""
    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", (ROOT / "mlruns").as_uri())
    )
    mlflow.set_experiment("MSML_DaudHidayatRamadhan")


def load_data():
    """Verify dataset hashes and split separation, then return train, test and lineage."""
    manifest = json.loads((DATA / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if hashlib.sha256((DATA / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Processed checksum mismatch: {name}")
    train, test = [pd.read_csv(DATA / f"{name}.csv.gz") for name in ("train", "test")]
    if not set(train.sample_id).isdisjoint(test.sample_id):
        raise ValueError("Train/test identity overlap")
    return train, test, manifest


def make_model(c=1.0):
    """Build an unfitted pipeline; each training fold fits its own text features."""
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 4),
        max_features=20000,
        min_df=2,
        dtype=np.float32,
        lowercase=False,
    )
    features = ColumnTransformer(
        [("text", vectorizer, "Sentence")], sparse_threshold=1.0
    )
    classifier = LinearSVC(C=c, dual="auto", max_iter=5000, random_state=42)
    return Pipeline([("features", features), ("classifier", classifier)])


def scores(truth, prediction, prefix, training=False):
    """Return weighted training metrics or explicitly named holdout metrics for logging parity."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy = float(accuracy_score(truth, prediction))
    if training:
        return {
            "training_accuracy_score": accuracy,
            "training_score": accuracy,
            "training_precision_score": float(
                precision_score(truth, prediction, average="weighted", zero_division=0)
            ),
            "training_recall_score": float(
                recall_score(truth, prediction, average="weighted", zero_division=0)
            ),
            "training_f1_score": float(
                f1_score(truth, prediction, average="weighted", zero_division=0)
            ),
        }
    return {
        prefix + "accuracy": accuracy,
        prefix
        + "macro_precision": float(
            precision_score(truth, prediction, average="macro", zero_division=0)
        ),
        prefix
        + "macro_recall": float(
            recall_score(truth, prediction, average="macro", zero_division=0)
        ),
        prefix
        + "macro_f1": float(
            f1_score(truth, prediction, average="macro", zero_division=0)
        ),
        prefix
        + "weighted_f1": float(
            f1_score(truth, prediction, average="weighted", zero_division=0)
        ),
    }
