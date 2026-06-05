#!/usr/bin/env python3
"""Train DeepResidual WITHOUT DASS features — for test set where DASS is unavailable.

Problem: Models trained with DASS (scores_das) get MF1 0.36 OOF but 0.09 on test
because DASS features are zero for test subjects → model collapses to majority class.

Solution: Train with dass_scheme="none", using only audio-video features.
"""

import argparse, copy, json, math, os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from ccac.baselines.anxiety_baseline import (
    BaselineConfig, _load_release_train_val, _apply_scaler, _fit_scaler,
    _build_folds, _class_weights, _classification_metrics, _set_seed,
    _resolve_device, _resolve_num_workers, _is_release_dataset, _extended_metrics,
)
from ccac.baselines.dass_baseline import (
    DASSConfig, FocalLoss, DASSDataset, _extract_dass_features,
    _calibrate_thresholds, _apply_thresholds,
)
from ccac.experiments.deep_residual import DeepResidualModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-path', default='datasets')
    p.add_argument('--output-dir', default='artifacts/exp/no_dass')
    p.add_argument('--audio-feature-name', default='audio_wavlm_base')
    p.add_argument('--video-feature-name', default='video_clip_base')
    p.add_argument('--device', default='cuda')
    p.add_argument('--num-folds', type=int, default=5)
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--patience', type=int, default=12)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--focal-gamma', type=float, default=1.0)
    p.add_argument('--class-weight-power', type=float, default=1.0)
    p.add_argument('--use-basic-features', action='store_true')
    args = p.parse_args()

    _set_seed(args.seed)
    device = _resolve_device(args.device)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Device: {device}, Output: {output_dir}')

    # Load data
    bc = BaselineConfig(
        dataset_path=str(Path(args.dataset_path).resolve()),
        output_dir='.', audio_feature_name=args.audio_feature_name,
        video_feature_name=args.video_feature_name,
    )
    frame, av_features, clip_mask, label_mapping, input_dim = _load_release_train_val(
        bc, Path(args.dataset_path))
    labels = frame['_label_index'].to_numpy(np.int64)

    # Optional: add basic features
    if args.use_basic_features:
        from ccac.baselines.dass_baseline import _load_multi_train_val
        audio_list = [args.audio_feature_name, 'audio_basic']
        video_list = [args.video_feature_name, 'video_basic']
        frame, av_features, clip_mask, input_dim = _load_multi_train_val(
            Path(args.dataset_path), frame, audio_list, video_list,
            bc.target_label_column, bc.feature_cache)
        print(f'With basic features: input_dim={input_dim}')

    print(f'Loaded {len(frame)} subjects, input_dim={input_dim}, classes={len(label_mapping)}')

    # NO DASS: use empty features
    dass_features = np.zeros((len(frame), 0), np.float32)

    fold_indices = _build_folds(labels, args.num_folds, args.seed)
    oof_probs = np.zeros((len(frame), len(label_mapping)), np.float32)
    oof_preds = np.full(len(frame), -1, np.int64)
    metrics = []

    for fold_id, (tr, vl) in enumerate(fold_indices, start=1):
        fold_dir = output_dir / f'fold_{fold_id}'
        fold_dir.mkdir(parents=True, exist_ok=True)
        workers = _resolve_num_workers(0)
        print(f'Fold {fold_id}/{args.num_folds}...', end=' ', flush=True)

        scaler = _fit_scaler(av_features[tr], clip_mask[tr])
        train_av = _apply_scaler(av_features[tr], scaler)
        val_av = _apply_scaler(av_features[vl], scaler)

        train_ds = DASSDataset(train_av, clip_mask[tr], dass_features[tr], labels[tr])
        val_ds = DASSDataset(val_av, clip_mask[vl], dass_features[vl], labels[vl])
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=workers)
        val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=workers)

        model = DeepResidualModel(
            input_dim=input_dim, num_classes=len(label_mapping),
            hidden_dim=256, num_heads=4, num_residual_blocks=3, dropout=0.2,
            dass_config=DASSConfig(dass_scheme='none'),
        ).to(device)

        cw = _class_weights(labels[tr], len(label_mapping), args.class_weight_power)
        if cw is not None: cw = cw.to(device)
        criterion = FocalLoss(gamma=args.focal_gamma, alpha=cw)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-6)

        best_score, best_state, patience_ct = -math.inf, None, 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            for av, m, d, lb in train_dl:
                av, m, lb = av.to(device), m.to(device), lb.to(device)
                opt.zero_grad(set_to_none=True)
                loss = criterion(model(av, m, d.to(device)), lb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()

            model.eval()
            vprobs_list, vlbls_list = [], []
            with torch.no_grad():
                for av, m, d, lb in val_dl:
                    av, m, lb = av.to(device), m.to(device), lb.to(device)
                    logits = model(av, m, d.to(device))
                    vprobs_list.append(torch.softmax(logits, -1).cpu().numpy())
                    vlbls_list.append(lb.cpu().numpy())
            vprob = np.concatenate(vprobs_list)
            vlbl = np.concatenate(vlbls_list)
            vpred = vprob.argmax(1)
            val_metrics = _extended_metrics(vlbl, vpred, vprob, len(label_mapping))
            selection_metric = 'robust_score'
            selection_score = float(val_metrics[selection_metric])

            if selection_score > best_score:
                best_score = selection_score
                patience_ct = 0
                best_state = {
                    'model': copy.deepcopy(model.state_dict()),
                    'scaler_mean': scaler[0].tolist(),
                    'scaler_std': scaler[1].tolist(),
                    'epoch': epoch,
                    'metrics': val_metrics,
                    'selection_metric': selection_metric,
                    'selection_score': selection_score,
                }
                best_probs = vprob
            else:
                patience_ct += 1
            if patience_ct >= args.patience:
                break

        model.load_state_dict(best_state['model'])
        oof_probs[vl] = best_probs
        oof_preds[vl] = best_probs.argmax(1)
        fold_metrics = _extended_metrics(labels[vl], oof_preds[vl], best_probs, len(label_mapping))
        print(
            f"MF1={fold_metrics['macro_f1']:.4f} "
            f"AUC={fold_metrics['macro_auc']:.4f} "
            f"Robust={fold_metrics['robust_score']:.4f} "
            f"Acc={fold_metrics['accuracy']:.4f}"
        )
        metrics.append({
            'fold': fold_id,
            'best_epoch': best_state.get('epoch'),
            'selection_metric': best_state.get('selection_metric', 'robust_score'),
            'selection_score': best_state.get('selection_score'),
            **fold_metrics,
        })
        torch.save(best_state, fold_dir / 'best_model.pt')

    overall = _extended_metrics(labels, oof_preds, oof_probs, len(label_mapping))
    print(
        f"\nOOF NO-DASS: MF1={overall['macro_f1']:.4f} "
        f"AUC={overall['macro_auc']:.4f} "
        f"Robust={overall['robust_score']:.4f} "
        f"Acc={overall['accuracy']:.4f}"
    )

    label_by_idx = {i: l for l, i in label_mapping.items()}
    print(classification_report(labels, oof_preds,
          target_names=[l for l,_ in sorted(label_mapping.items(), key=lambda x:x[1])],
          zero_division=0))

    # Save OOF
    oof_df = pd.DataFrame({
        'subject_id': frame['subject_id'].astype(str),
        'true_label': frame['t4_anxiety_level'].astype(str),
        'pred_label': [label_by_idx[int(i)] for i in oof_preds],
    })
    for ci in range(len(label_mapping)):
        oof_df[f'prob_class_{ci}'] = oof_probs[:, ci]
    oof_df.to_csv(output_dir / 'oof_predictions.csv', index=False)
    pd.DataFrame(metrics).to_csv(output_dir / 'fold_metrics.csv', index=False)
    (output_dir / 'label_mapping.json').write_text(json.dumps(label_mapping, ensure_ascii=False, indent=2))

    summary = {
        'selection_metric': 'robust_score',
        'overall_oof_metrics': overall,
        'oof_mf1': overall['macro_f1'],
        'oof_acc': overall['accuracy'],
        'oof_macro_auc': overall['macro_auc'],
        'oof_robust_score': overall['robust_score'],
        'input_dim': input_dim,
        'label_mapping': label_mapping,
        'dass_scheme': 'none',
        'focal_gamma': args.focal_gamma,
        'class_weight_power': args.class_weight_power,
        'use_basic_features': args.use_basic_features,
    }
    (output_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Done. Saved to {output_dir}')


if __name__ == '__main__':
    main()
