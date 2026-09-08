"""Verify the complete exported model and its deterministic inference fixture."""
import hashlib
import json
from pathlib import Path
import sys
import mlflow.pyfunc
import pandas as pd

root = Path(sys.argv[1] if len(sys.argv) > 1 else 'output')
manifest = json.loads((root/'manifest.json').read_text())
assert manifest['status'] == 'FINISHED'
for name, expected in manifest['files'].items():
    path = (root/name).resolve()
    assert path.is_relative_to(root.resolve()), 'Manifest path escapes artifact root'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
fixture = json.loads((root/'inference_fixture.json').read_text())
split = fixture['dataframe_split']
prediction = mlflow.pyfunc.load_model(str(root/'model')).predict(pd.DataFrame(split['data'], columns=split['columns'])).tolist()
assert len(prediction) == len(fixture['expected']) and prediction == fixture['expected']
print(json.dumps({'source_commit': manifest['source_commit'], 'run_id': manifest['run_id'], 'hashes': 'pass', 'batch_predictions': 'pass', 'rows': len(prediction)}))
