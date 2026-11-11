"""Check that the standalone program built by ``scripts/build_standalone.py`` really works on this computer.

    python scripts/smoke_standalone.py

1. analyses a sample sales file with ``DemandPlanner --analyse`` (forecasts, policies, saved results);
2. starts the dashboard and waits until it answers.
Uses a temporary company folder; exits with an error when a step fails.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "dist" / "DemandPlanner" / ("DemandPlanner.exe" if os.name == "nt" else "DemandPlanner")
SAMPLE = ROOT / "data" / "samples" / "single_store_it.csv"
PORT = 8599
ANALYSIS_TIMEOUT, START_TIMEOUT = 900, 240


def answers(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/_stcore/health", timeout=2) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def main() -> int:
    if not PROGRAM.exists():
        print(f"Not built yet: {PROGRAM}", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        folder = Path(tmp) / "Company"
        (folder / "data").mkdir(parents=True)
        shutil.copy(SAMPLE, folder / "data" / SAMPLE.name)
        env = os.environ | {"PLANNER_WORKSPACE": str(folder), "PLANNER_NO_BROWSER": "1", "PLANNER_PORT": str(PORT)}

        print("1/2 analysis without the dashboard…", flush=True)
        subprocess.run([str(PROGRAM), "--analyse"], env=env, check=True, timeout=ANALYSIS_TIMEOUT)
        plan = folder / "results" / "reorder_parameters.csv"
        if not plan.exists() or len(plan.read_text(encoding="utf-8").splitlines()) < 2:
            print("The analysis wrote no reorder parameters.", file=sys.stderr)
            return 1

        print("2/2 dashboard…", flush=True)
        process = subprocess.Popen([str(PROGRAM)], env=env)
        try:
            deadline = time.time() + START_TIMEOUT
            while not answers(PORT):
                if process.poll() is not None:
                    print(f"The dashboard stopped while starting (code {process.returncode}).", file=sys.stderr)
                    return 1
                if time.time() > deadline:
                    print("The dashboard did not answer in time.", file=sys.stderr)
                    return 1
                time.sleep(1)
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
    print("OK: the standalone program analyses a file and serves the dashboard.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
