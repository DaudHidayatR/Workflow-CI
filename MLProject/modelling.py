"""Retrain the previously selected configuration; export a portable CI artifact."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import numpy as np
import mlflow
import mlflow.sklearn
from model_support import configure, load_data, make_model, scores


def main():
    """Train the fixed model, verify saved predictions, and record metrics and artifact evidence."""
    configure()
    train, test, data_manifest = load_data()
    output = Path(os.environ.get("OUTPUT_DIR", "../output")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_dir = output / "model"
    if model_dir.exists():
        raise RuntimeError("Output model already exists; use a fresh output directory")
    mlflow.sklearn.autolog(disable=True)
    # C=2 was selected before CI using train-only cross-validation in iteration 7.
    model = make_model(c=2.0)
    with mlflow.start_run() as active:
        mlflow.set_tags(
            {
                "stage": "ci-retraining",
                "student": "DaudHidayatRamadhan",
                "source_commit": os.environ.get("SOURCE_COMMIT", "local-preflight"),
                "selection": "fixed C=2 from iteration-07 train-only CV",
            }
        )
        model.fit(train[["Sentence"]], train.target)
        metrics = {
            **scores(
                train.target, model.predict(train[["Sentence"]]), "", training=True
            ),
            **scores(test.target, model.predict(test[["Sentence"]]), "holdout_"),
        }
        mlflow.log_params({k: str(v) for k, v in model.get_params(deep=True).items()})
        mlflow.log_metrics(metrics)
        signature = mlflow.models.infer_signature(
            test[["Sentence"]].iloc[:3], model.predict(test[["Sentence"]].iloc[:3])
        )
        mlflow.sklearn.log_model(model, "model", signature=signature)
        mlflow.log_dict(data_manifest, "processed_manifest.json")
        run_id = active.info.run_id
    downloaded = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="model"
    )
    shutil.copytree(downloaded, model_dir)
    sample = test[["Sentence"]].iloc[:32]
    expected = model.predict(sample).tolist()
    generic = mlflow.pyfunc.load_model(str(model_dir))
    actual = generic.predict(sample).tolist()
    if not (len(actual) == 32 and actual == expected):
        raise AssertionError("Exported model prediction mismatch")
    if not (
        np.array_equal(
            mlflow.sklearn.load_model(str(model_dir)).predict(sample), expected
        )
    ):
        raise AssertionError(
            "Verification failed: np.array_equal(mlflow.sklearn.load_model(str(model_dir)).predict(sample), expected)"
        )
    run = mlflow.get_run(run_id)
    if not (run.info.status == "FINISHED"):
        raise AssertionError("Verification failed: run.info.status == 'FINISHED'")
    for key, value in metrics.items():
        if not (np.isclose(run.data.metrics[key], value)):
            raise AssertionError(key)
    (output / "inference_fixture.json").write_text(
        json.dumps(
            {"dataframe_split": sample.to_dict(orient="split"), "expected": expected},
            indent=2,
        )
        + "\n"
    )
    (output / "processed_manifest.json").write_text(
        json.dumps(data_manifest, indent=2) + "\n"
    )
    manifest = {
        "run_id": run_id,
        "status": run.info.status,
        "source_commit": os.environ.get("SOURCE_COMMIT", "local-preflight"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID", ""),
        "metrics": metrics,
        "data_sha256": data_manifest["raw_sha256"],
        "selected_C": 2.0,
        "pyfunc_roundtrip": "pass",
        "files": {
            str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
