"""Prepare MLflow's descriptor from the submission-named source, then train."""
from pathlib import Path
import os
import shutil


def prepare_project(project: Path) -> Path:
    source = project / 'MLProject'
    target = project / 'MLproject'
    # On case-insensitive filesystems these names already identify one file.
    if not target.exists() or not source.samefile(target):
        shutil.copyfile(source, target)
    return target


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    prepare_project(root / 'MLProject')
    os.chdir(root)
    os.execvp('mlflow', ['mlflow', 'run', 'MLProject', '--env-manager', 'local',
                        '--experiment-name', 'MSML_DaudHidayatRamadhan'])
