"""Verify the complete exported model and its deterministic inference fixture."""

import hashlib
import json
from pathlib import Path
import sys
import mlflow.pyfunc
import pandas as pd


def main():
    """Run this command explicitly; importing the module performs no work."""
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "output")
    manifest = json.loads((root / "manifest.json").read_text())
    if not (manifest["status"] == "FINISHED"):
        raise AssertionError("Verification failed: manifest['status'] == 'FINISHED'")
    for name, expected in manifest["files"].items():
        path = (root / name).resolve()
        if not (path.is_relative_to(root.resolve())):
            raise AssertionError("Manifest path escapes artifact root")
        if not (hashlib.sha256(path.read_bytes()).hexdigest() == expected):
            raise AssertionError(name)
    fixture = json.loads((root / "inference_fixture.json").read_text())
    split = fixture["dataframe_split"]
    prediction = (
        mlflow.pyfunc.load_model(str(root / "model"))
        .predict(pd.DataFrame(split["data"], columns=split["columns"]))
        .tolist()
    )
    if not (
        len(prediction) == len(fixture["expected"])
        and prediction == fixture["expected"]
    ):
        raise AssertionError(
            "Verification failed: len(prediction) == len(fixture['expected']) and prediction == fixture['expected']"
        )
    print(
        json.dumps(
            {
                "source_commit": manifest["source_commit"],
                "run_id": manifest["run_id"],
                "hashes": "pass",
                "batch_predictions": "pass",
                "rows": len(prediction),
            }
        )
    )


if __name__ == "__main__":
    main()
