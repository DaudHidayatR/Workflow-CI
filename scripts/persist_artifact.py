"""Commit each verified CI artifact to the same repository's artifact branch."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def git(*args, cwd=None, capture=False):
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )


repo = os.environ["GITHUB_REPOSITORY"]
run_key = os.environ["GITHUB_RUN_ID"] + "-" + os.environ.get("GITHUB_RUN_ATTEMPT", "1")
source = Path("output").resolve()
manifest = json.loads((source / "manifest.json").read_text())
assert manifest["source_commit"] == os.environ["GITHUB_SHA"]
with tempfile.TemporaryDirectory(prefix="msml-artifact-") as temp:
    root = Path(temp)
    git("init", "-b", "model-artifacts", cwd=root)
    git("config", "user.name", "github-actions[bot]", cwd=root)
    git(
        "config",
        "user.email",
        "41898282+github-actions[bot]@users.noreply.github.com",
        cwd=root,
    )
    git("config", "credential.helper", "!gh auth git-credential", cwd=root)
    git("remote", "add", "origin", f"https://github.com/{repo}.git", cwd=root)
    exists = git(
        "ls-remote", "--heads", "origin", "model-artifacts", cwd=root, capture=True
    ).stdout.strip()
    if exists:
        git("fetch", "--depth=1", "origin", "model-artifacts", cwd=root)
        git("reset", "--soft", "FETCH_HEAD", cwd=root)
        git("checkout", "FETCH_HEAD", "--", ".", cwd=root)
    destination = root / "models" / run_key
    shutil.copytree(source, destination)
    (root / "latest.json").write_text(
        json.dumps(
            {
                "path": f"models/{run_key}",
                "source_commit": manifest["source_commit"],
                "mlflow_run_id": manifest["run_id"],
                "github_run_id": os.environ["GITHUB_RUN_ID"],
            },
            indent=2,
        )
        + "\n"
    )
    (root / "README.md").write_text(
        "# Verified CI model artifacts\n\nEach models directory is a distinct CI run. latest.json points to the most recently persisted successful run. Verify manifest.json hashes before inference. Source code is on main.\n"
    )
    git("add", ".", cwd=root)
    git("commit", "-m", f"Persist verified model from CI {run_key}", cwd=root)
    git("push", "origin", "HEAD:refs/heads/model-artifacts", cwd=root)
