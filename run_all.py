"""
run_all.py

Runs the whole project in the right order with one command:

    python run_all.py              # everything, from raw data to final evaluation
    python run_all.py --skip-data  # reuse existing data/processed files and
                                   # start at model training

Order matters:
  - anomaly_detection.py rewrites the processed CSVs, so it must run after
    data_pipeline.py.
  - train_baseline.py must run before anything that loads models/rf_model.pkl.
  - evaluate_test.py is the final exam; run it last.

The script stops at the first failure, so you never get results from a
half-finished chain.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

DATA_STEPS = ["data_pipeline.py", "anomaly_detection.py"]
MODEL_STEPS = [
    "train_baseline.py",
    "explainability.py",
    "generate_review_queue.py",
    "evaluate_test.py",
]


def run(script: str) -> None:
    path = SRC / script
    if not path.exists():
        sys.exit(f"Missing {path}. Check the file name (no spaces) and location.")
    print(f"\n{'=' * 64}\nRunning {script}\n{'=' * 64}", flush=True)
    start = time.time()
    result = subprocess.run([sys.executable, str(path)], cwd=ROOT)
    if result.returncode != 0:
        sys.exit(f"\n{script} failed (exit code {result.returncode}). Stopping.")
    print(f"\n{script} finished in {time.time() - start:.0f}s", flush=True)


def main() -> None:
    steps = MODEL_STEPS if "--skip-data" in sys.argv else DATA_STEPS + MODEL_STEPS
    if not (SRC / "risk_model.py").exists():
        sys.exit("src/risk_model.py not found. It must be named exactly risk_model.py.")
    for script in steps:
        run(script)
    print("\nAll steps completed.")


if __name__ == "__main__":
    main()
