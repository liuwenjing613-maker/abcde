#!/usr/bin/env python3
"""Run v4a-v4d loss ablations (CE only / CE+KD / CE+ordinal / full)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


MODES = {
    "v4a": ("ce_only", 0.0, 0.0),
    "v4b": ("ce_kd", 0.5, 0.0),
    "v4c": ("ce_ordinal", 0.0, 0.2),
    "v4d": ("ce_kd_ordinal", 0.5, 0.2),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v4.1 loss ablations")
    parser.add_argument("--dataset-path", type=str, default="/home/adodas/dataset_ccac")
    parser.add_argument("--output-root", type=str, default="artifacts/ablations")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--modes", nargs="+", default=list(MODES.keys()))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    results = {}
    for name in args.modes:
        loss_mode, kd_w, ord_w = MODES[name]
        out_dir = Path(args.output_root) / name
        cmd = [
            sys.executable,
            str(root / "scripts" / "train_stagewise_v41.py"),
            "--dataset-path",
            args.dataset_path,
            "--output-dir",
            str(out_dir),
            "--loss-mode",
            loss_mode,
            "--kd-weight",
            str(kd_w),
            "--ordinal-weight",
            str(ord_w),
            "--device",
            args.device,
        ]
        print("Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=str(root), env={**dict(__import__("os").environ), "PYTHONPATH": str(root / "src")}, check=True)
        summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
        results[name] = summary["overall_oof_metrics"]

    report_path = Path(args.output_root) / "ablation_summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
