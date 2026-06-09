#!/usr/bin/env python3
"""Run full v4.1 pipeline in order: export -> fusion -> v41 -> ablations -> shallow mm."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], root: Path) -> None:
    print(">>>", " ".join(cmd), flush=True)
    subprocess.run(
        cmd,
        cwd=str(root),
        env={**dict(__import__("os").environ), "PYTHONPATH": str(root / "src")},
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ordered v4.1 experiment pipeline")
    parser.add_argument("--dataset-path", type=str, default="/home/adodas/dataset_ccac")
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    py = sys.executable

    run([py, str(root / "scripts" / "export_baseline_oof.py"), "--dataset-path", args.dataset_path], root)
    run([py, str(root / "scripts" / "fusion_baseline_v3.py"), "--dataset-path", args.dataset_path], root)
    run([
        py, str(root / "scripts" / "train_stagewise_v41.py"),
        "--dataset-path", args.dataset_path,
        "--output-dir", "artifacts/stagewise_v41",
        "--device", args.device,
    ], root)
    if not args.skip_ablations:
        run([
            py, str(root / "scripts" / "run_v41_ablations.py"),
            "--dataset-path", args.dataset_path,
            "--device", args.device,
        ], root)
    run([py, str(root / "scripts" / "train_shallow_mm.py"), "--dataset-path", args.dataset_path], root)


if __name__ == "__main__":
    main()
