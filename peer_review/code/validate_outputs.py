#!/usr/bin/env python3
"""
validate_outputs.py
===================
Run the validation harness for current replication artifacts.

By default, this runs artifact/package checks against the existing outputs.
Use --full-pipeline to rebuild outputs first via run_all.py.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PEER_REVIEW_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = PEER_REVIEW_DIR.parent
TESTS_DIR = PROJECT_ROOT / "tests"


def run_command(command: list[str], cwd: Path) -> int:
    """Execute a subprocess and stream output to the terminal."""
    result = subprocess.run(command, cwd=cwd)
    return result.returncode


def main() -> int:
    """Optionally rebuild outputs, then run the validation suite."""
    parser = argparse.ArgumentParser(description="Validate the shipped replication outputs.")
    parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Run peer_review/code/run_all.py before executing the validation tests.",
    )
    args = parser.parse_args()

    if args.full_pipeline:
        pipeline_cmd = [sys.executable, str(SCRIPT_DIR / "run_all.py")]
        pipeline_code = run_command(pipeline_cmd, cwd=PEER_REVIEW_DIR)
        if pipeline_code != 0:
            return pipeline_code

    test_cmd = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        str(TESTS_DIR),
        "-v",
    ]
    return run_command(test_cmd, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
