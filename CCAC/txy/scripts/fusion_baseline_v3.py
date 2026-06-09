#!/usr/bin/env python3
"""
Step 1-3 fusion pipeline:
  1) baseline + v3 weight search on OOF
  2) generate 3 fixed-weight submission zips
  3) baseline + v3 + small v4 blend + shrink/prior variants
"""
from __future__ import annotations

import argparse
import json
import zipfile
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from txy.constants import INDEX_TO_LEVEL, LEVEL_TO_INDEX, SUBMISSION_LEVEL_TO_INDEX
from txy.data.logit_io import build_train_subject_order, load_oof_aligned, load_test_logits_aligned
from txy.data.labels import encode_ordinal_labels
from txy.ensemble.class_wise import blend_logits
from txy.training.calibration import apply_class_bias, search_class_bias
from txy.training.metrics import classification_metrics
from txy.training.prior_calibration import apply_prior_bias, compute_train_prior_bias


def _softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)


def _write_submission_zip(
    test_frame: pd.DataFrame,
    logits: np.ndarray,
    output_zip: Path,
    csv_name: str = "submission.csv",
) -> dict:
    preds = logits.argmax(axis=1)
    submission = test_frame[["anon_school", "anon_class", "anon_person"]].copy()
    submission["label"] = [int(p) for p in preds]
    staging = output_zip.parent / f"_staging_{output_zip.stem}"
    staging.mkdir(parents=True, exist_ok=True)
    csv_path = staging / csv_name
    submission.to_csv(csv_path, index=False, encoding="utf-8")
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname=csv_name)
    counts = submission["label"].value_counts().sort_index().to_dict()
    return {
        "zip": str(output_zip),
        "label_counts": {int(k): int(v) for k, v in counts.items()},
        "label_named": {INDEX_TO_LEVEL[int(k)]: int(v) for k, v in counts.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline+v3 fusion and submission generation")
    parser.add_argument("--dataset-path", type=str, default="/home/adodas/dataset_ccac")
    parser.add_argument("--baseline-oof", type=str, default="artifacts/baseline_ordinal/oof_predictions.csv")
    parser.add_argument("--baseline-test", type=str, default="artifacts/baseline_ordinal/test_predictions.csv")
    parser.add_argument("--v3-oof", type=str, default="artifacts/residual_v3/oof_predictions.csv")
    parser.add_argument("--v3-test", type=str, default="artifacts/residual_v3/test_predictions_submission.csv")
    parser.add_argument("--v4-oof", type=str, default="artifacts/stagewise_v4/oof_predictions.csv")
    parser.add_argument("--v4-test", type=str, default="artifacts/stagewise_v4/test_predictions_submission.csv")
    parser.add_argument("--expected-rows", type=int, default=382)
    parser.add_argument("--output-dir", type=str, default="artifacts/fusion")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    frame, subject_ids = build_train_subject_order(dataset_root)
    labels, _ = encode_ordinal_labels(frame["t4_anxiety_level"])

    _, baseline_oof = load_oof_aligned(Path(args.baseline_oof), subject_ids)
    _, v3_oof = load_oof_aligned(Path(args.v3_oof), subject_ids)

    # --- Step 1: weight search baseline + v3 ---
    weight_grid = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    search_results: list[dict] = []
    best = {"macro_f1": -1.0}
    for w_v3 in weight_grid:
        w_base = 1.0 - w_v3
        blended = blend_logits([baseline_oof, v3_oof], [w_base, w_v3])
        bias, metrics = search_class_bias(blended, labels)
        record = {
            "w_v3": w_v3,
            "w_baseline": w_base,
            "class_bias": bias.tolist(),
            **metrics,
        }
        search_results.append(record)
        if metrics["macro_f1"] > best.get("macro_f1", -1):
            best = record

    (output_dir / "baseline_v3_search.json").write_text(
        json.dumps({"best": best, "all": search_results}, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    test_full = pd.read_csv(dataset_root / "test" / "subjects.csv")
    test_frame = test_full.iloc[: args.expected_rows].copy()
    baseline_test = load_test_logits_aligned(Path(args.baseline_test), test_frame)
    v3_test = load_test_logits_aligned(Path(args.v3_test), test_frame)

    # --- Step 2: three fixed-weight submissions ---
    fixed_weights = [(0.7, 0.3), (0.5, 0.5), (0.8, 0.2)]
    fixed_reports = []
    for w_v3, w_base in fixed_weights:
        logits = blend_logits([baseline_test, v3_test], [w_base, w_v3])
        # OOF bias from same weights
        oof_blend = blend_logits([baseline_oof, v3_oof], [w_base, w_v3])
        bias, _ = search_class_bias(oof_blend, labels)
        logits = apply_class_bias(logits, bias)
        zip_path = Path(__file__).resolve().parents[1] / "submissions" / f"fusion_baseline_v3_w{int(w_v3*10)}.zip"
        report = _write_submission_zip(test_frame, logits, zip_path)
        report.update({"w_v3": w_v3, "w_baseline": w_base, "bias": bias.tolist()})
        fixed_reports.append(report)

    # --- Step 3: baseline + v3 + 0.1*v4 with calibration variants ---
    v4_reports = []
    if Path(args.v4_test).exists() and Path(args.v4_oof).exists():
        _, v4_oof = load_oof_aligned(Path(args.v4_oof), subject_ids)
        v4_test = load_test_logits_aligned(Path(args.v4_test), test_frame)
        for w_v3 in [0.7, 0.5]:
            w_v4 = 0.1
            w_base = 1.0 - w_v3 - w_v4
            oof_blend = blend_logits([baseline_oof, v3_oof, v4_oof], [w_base, w_v3, w_v4])
            test_blend = blend_logits([baseline_test, v3_test, v4_test], [w_base, w_v3, w_v4])

            calibrations = {
                "oof_bias": lambda lg: apply_class_bias(lg, search_class_bias(oof_blend, labels)[0]),
                "shrink_bias": lambda lg: apply_class_bias(
                    lg, 0.5 * search_class_bias(oof_blend, labels)[0]
                ),
                "prior": lambda lg: apply_prior_bias(
                    lg, compute_train_prior_bias(frame["t4_anxiety_level"], LEVEL_TO_INDEX, 5)
                ),
            }
            for name, fn in calibrations.items():
                logits = fn(test_blend.copy())
                zip_path = (
                    Path(__file__).resolve().parents[1]
                    / "submissions"
                    / f"fusion_baseline_v3_v4_w{int(w_v3*10)}_{name}.zip"
                )
                report = _write_submission_zip(test_frame, logits, zip_path)
                report.update({"w_v3": w_v3, "w_v4": w_v4, "w_baseline": w_base, "calibration": name})
                v4_reports.append(report)

    summary = {
        "oof_best": best,
        "fixed_submissions": fixed_reports,
        "v4_submissions": v4_reports,
    }
    (output_dir / f"fusion_summary_{stamp}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
