"""Exercise a loopback MLflow endpoint using the verified CI fixture."""
import argparse
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--artifact', type=Path, default=Path('output'))
    parser.add_argument('--output', type=Path, default=Path('serving-proof.json'))
    args = parser.parse_args()
    fixture = json.loads((args.artifact/'inference_fixture.json').read_text())
    manifest = json.loads((args.artifact/'manifest.json').read_text())
    for attempt in range(60):
        try:
            with urlopen(args.url+'/ping', timeout=2) as response:
                if response.status == 200: break
        except (URLError, TimeoutError):
            if attempt == 59: raise
            time.sleep(1)
    started = time.perf_counter()
    request = Request(args.url+'/invocations', data=json.dumps({'dataframe_split': fixture['dataframe_split']}).encode(), headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=60) as response:
        result = json.load(response)
    assert result['predictions'] == fixture['expected'], 'HTTP / CI model predictions differ'
    bad = Request(args.url+'/invocations', data=b'{"dataframe_records":[{"wrong_column":"example"}]}', headers={'Content-Type':'application/json'})
    try:
        urlopen(bad, timeout=10)
    except HTTPError as error:
        assert error.code == 400, error.code
        rejected = error.code
    else:
        raise AssertionError('Missing Sentence column was accepted')
    proof = {'source_commit': manifest['source_commit'], 'mlflow_run_id': manifest['run_id'],
             'url': args.url, 'rows': len(result['predictions']), 'predictions_match_ci': True,
             'invalid_schema_status': rejected, 'elapsed_seconds': time.perf_counter()-started,
             'predictions': result['predictions']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps(proof, indent=2))

if __name__ == '__main__':
    main()
