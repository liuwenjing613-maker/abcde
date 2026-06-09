#!/usr/bin/env python3
"""Search late-fusion weights on OOF logits (tabular + stagewise + optional baseline)."""
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from txy.constants import INDEX_TO_LEVEL, LEVEL_TO_INDEX, TARGET_LABEL_COLUMN
from txy.data.labels import encode_ordinal_labels
from txy.training.calibration import apply_class_bias, search_class_bias
from txy.training.metrics import classification_metrics


def load_logits_csv(path: Path, num_classes: int = 5) -> np.ndarray:
    df = pd.read_csv(path)
    logit_cols = [f"logit_class_{i}" for i in range(num_classes)]
    if all(c in df.columns for c in logit_cols):
        return df[logit_cols].to_numpy(dtype=np.float32)
    prob_cols = [f"prob_class_{i}" for i in range(num_classes)]
    probs = df[prob_cols].to_numpy(dtype=np.float32)
    return np.log(probs.clip(1e-6, 1.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search late fusion weights on OOF")
    parser.add_argument("--dataset-path", type=str, default="/home/adodas/dataset_ccac")
    parser.add_argument("--tabular-oof", type=str, default="artifacts/history_tabular/oof_predictions.csv")
    parser.add_argument("--stagewise-oof", type=str, default=None)
    parser.add_argument("--residual-v3-oof", type=str, default="artifacts/residual_v3/oof_predictions.csv")
    parser.add_argument("--output", type=str, default="artifacts/late_fusion/weight_search.json")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_path)
    frame = pd.read_csv(dataset_root / "train_val" / "labels.csv").dropna(subset=[TARGET_LABEL_COLUMN])
    labels, _ = encode_ordinal_labels(frame[TARGET_LABEL_COLUMN])

    sources: dict[str, np.ndarray] = {}
    if args.tabular_oof:
        sources["tabular"] = load_logits_csv(Path(args.tabular_oof))
    if args.stagewise_oof:
        sources["stagewise"] = load_logits_csv(Path(args.stagewise_oof))
    if args.residual_v3_oof:
        sources["residual_v3"] = load_logits_csv(Path(args.residual_v3_oof))

    if len(sources) < 2:
        raise ValueError("need at least two OOF logit sources")

    n = len(labels)
    for name, logits in sources.items():
        if logits.shape[0] != n:
            raise ValueError(f"{name} has {logits.shape[0]} rows, expected {n}")

    weight_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    names = list(sources.keys())
    best = {"macro_f1": -1.0}
    results: list[dict] = []

    if len(names) == 2:
        grids = product(weight_grid, repeat=2)
        for w0, w1 in grids:
            if abs(w0 + w1 - 1.0) > 1e-6:
                continue
            blended = w0 * sources[names[0]] + w1 * sources[names[1]]
            bias, metrics = search_class_bias(blended, labels)
            record = {
                "weights": {names[0]: w0, names[1]: w1},
                "class_bias": bias.tolist(),
                **metrics,
            }
            results.append(record)
            if metrics["macro_f1"] > best.get("macro_f1", -1):
                best = record
    else:
        for w_tab in [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]:
            for w_mm in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
                w_res = 1.0 - w_tab - w_mm
                if w_res < -1e-6:
                    continue
                blended = w_tab * sources.get("tabular", 0) + w_mm * sources.get("stagewise", 0)
                if "residual_v3" in sources:
                    blended = blended + w_res * sources["residual_v3"]
                bias, metrics = search_class_bias(blended, labels)
                record = {
                    "weights": {"tabular": w_tab, "stagewise": w_mm, "residual_v3": max(w_res, 0.0)},
                    "class_bias": bias.tolist(),
                    **metrics,
                }
                results.append(record)
                if metrics["macro_f1"] > best.get("macro_f1", -1):
                    best = record

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"best": best, "top10": sorted(results, key=lambda r: r["macro_f1"], reverse=True)[:10]}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["best"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
