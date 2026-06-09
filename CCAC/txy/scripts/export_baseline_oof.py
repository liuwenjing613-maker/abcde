#!/usr/bin/env python3
"""Export official baseline OOF/test probs remapped to DASS ordinal logits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from txy.constants import INDEX_TO_LEVEL, SUBMISSION_LEVEL_TO_INDEX
from txy.data.logit_io import build_train_subject_order, load_oof_aligned, load_test_logits_aligned
from txy.data.feature_io import make_subject_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Export baseline predictions in ordinal logit format")
    parser.add_argument("--dataset-path", type=str, default="/home/adodas/dataset_ccac")
    parser.add_argument(
        "--baseline-artifact-dir",
        type=str,
        default="/home/adodas/CCAC/CCAC_MAPS/artifacts/baselines/anxiety_wavlm_dinov2_small",
    )
    parser.add_argument("--output-dir", type=str, default="artifacts/baseline_ordinal")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_path).resolve()
    baseline_dir = Path(args.baseline_artifact_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping = json.loads((baseline_dir / "label_mapping.json").read_text(encoding="utf-8"))
    frame, subject_ids = build_train_subject_order(dataset_root)

    oof_df, oof_logits = load_oof_aligned(
        baseline_dir / "oof_predictions.csv",
        subject_ids,
        baseline_dir / "label_mapping.json",
    )
    oof_out = frame[["anon_school", "anon_class", "anon_person", "subject_id", "t4_anxiety_level"]].copy()
    oof_out["true_label"] = frame["t4_anxiety_level"].astype(str)
    oof_out["pred_label"] = [INDEX_TO_LEVEL[int(i)] for i in oof_logits.argmax(axis=1)]
    for i in range(5):
        oof_out[f"logit_class_{i}"] = oof_logits[:, i]
        probs = np.exp(oof_logits - oof_logits.max(axis=1, keepdims=True))
        probs = probs / probs.sum(axis=1, keepdims=True)
        oof_out[f"prob_class_{i}"] = probs[:, i]
    oof_out.to_csv(output_dir / "oof_predictions.csv", index=False, encoding="utf-8")

    test_frame = pd.read_csv(dataset_root / "test" / "subjects.csv")
    test_frame["subject_id"] = make_subject_id(test_frame)
    test_logits = load_test_logits_aligned(
        baseline_dir / "test_predictions.csv",
        test_frame,
        baseline_dir / "label_mapping.json",
    )
    test_out = test_frame[["anon_school", "anon_class", "anon_person"]].copy()
    test_out["label"] = [INDEX_TO_LEVEL[int(i)] for i in test_logits.argmax(axis=1)]
    for i in range(5):
        test_out[f"prob_class_{i}"] = (
            np.exp(test_logits - test_logits.max(axis=1, keepdims=True))
            / np.exp(test_logits - test_logits.max(axis=1, keepdims=True)).sum(axis=1, keepdims=True)
        )[:, i]
    test_out.to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8")

    (output_dir / "label_mapping.json").write_text(
        json.dumps(SUBMISSION_LEVEL_TO_INDEX, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    meta = {
        "source": str(baseline_dir),
        "source_label_mapping": mapping,
        "ordinal_mapping": SUBMISSION_LEVEL_TO_INDEX,
        "oof_rows": int(len(oof_out)),
        "test_rows": int(len(test_out)),
    }
    (output_dir / "export_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
