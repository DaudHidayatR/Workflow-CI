# Workflow-CI — DaudHidayatRamadhan

This public repository retrains the four-class payload benchmark using the fixed C=2 configuration selected by training-only cross-validation. It preserves the previously verified 19,200/4,800 split and never searches using holdout performance.

## Run with Docker Compose

```bash
docker compose build retrain
docker compose run --rm retrain
```

On the author's rootless Podman workstation, prefix with `LOCAL_UID=0 LOCAL_GID=0`. Use a fresh output directory for each retraining; existing output/model is deliberately rejected. The container runs `mlflow run MLProject --env-manager local`, so the project's entry point executes inside the pinned Docker environment. `MLProject/conda.yaml` is retained for rubric compatibility; `MLProject/MLproject` is MLflow's case-sensitive executable descriptor, and `MLProject/MLProject` is an identical rubric-named copy.

## CI and durable artifact retrieval

Pushes to main affecting code/data/workflow and manual dispatch trigger training. The workflow verifies exported model hashes and 32 native/pyfunc predictions, commits output under `models/<run-id>-<attempt>` on this repository's `model-artifacts` branch, and uploads an Actions artifact. `latest.json` on that branch records the source commit and run. Download that directory, verify manifest hashes, and use its model and inference_fixture.json together.

The image is built using **mlflow models build-docker**, then checked via HTTP batch parity and malformed-schema rejection. Docker Hub publication requires repository variable `DOCKERHUB_USERNAME` and secret `DOCKERHUB_TOKEN` (write access to the intended namespace). Missing credentials explicitly leave publication incomplete; a successful training/build alone does not satisfy that exit gate. Published tags use the full source commit.

```bash
python scripts/verify_artifact.py output
python scripts/inference.py --artifact output --url http://127.0.0.1:8000
```

Dataset source: https://www.kaggle.com/datasets/alextrinity/sqli-xss-dataset version 2, publisher-declared Apache-2.0. Dataset/attribution files accompany MLProject. Payloads are inert text, never executed. Known limitations include publisher label quality, near-duplicate families and bounded sampling. This benchmark is not production security validation.
