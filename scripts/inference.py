"""Exercise a loopback MLflow endpoint using the verified CI fixture."""

import argparse
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


def wait_ready(url, attempts=60, interval=1):
    """Wait for HTTP 200, otherwise fail after a bounded number of probes."""
    for attempt in range(attempts):
        try:
            with urlopen(url + "/ping", timeout=2) as response:
                if response.status == 200:
                    return
        except (URLError, OSError):
            pass
        if attempt + 1 < attempts:
            time.sleep(interval)
    raise TimeoutError(f"Model did not become ready after {attempts} probes: {url}")


def main():
    """Check endpoint predictions against the CI fixture and save HTTP verification evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--artifact", type=Path, default=Path("output"))
    parser.add_argument("--output", type=Path, default=Path("serving-proof.json"))
    args = parser.parse_args()
    fixture = json.loads((args.artifact / "inference_fixture.json").read_text())
    manifest = json.loads((args.artifact / "manifest.json").read_text())
    wait_ready(args.url)
    started = time.perf_counter()
    request = Request(
        args.url + "/invocations",
        data=json.dumps({"dataframe_split": fixture["dataframe_split"]}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=60) as response:
        result = json.load(response)
    if not (result["predictions"] == fixture["expected"]):
        raise AssertionError("HTTP / CI model predictions differ")
    bad = Request(
        args.url + "/invocations",
        data=b'{"dataframe_records":[{"wrong_column":"example"}]}',
        headers={"Content-Type": "application/json"},
    )
    try:
        urlopen(bad, timeout=10)
    except HTTPError as error:
        if not (error.code == 400):
            raise AssertionError(error.code)
        rejected = error.code
    else:
        raise AssertionError("Missing Sentence column was accepted")
    proof = {
        "source_commit": manifest["source_commit"],
        "mlflow_run_id": manifest["run_id"],
        "url": args.url,
        "rows": len(result["predictions"]),
        "predictions_match_ci": True,
        "invalid_schema_status": rejected,
        "elapsed_seconds": time.perf_counter() - started,
        "predictions": result["predictions"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(proof, indent=2))


if __name__ == "__main__":
    main()
