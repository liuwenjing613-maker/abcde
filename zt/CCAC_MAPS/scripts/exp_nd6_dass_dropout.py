#!/usr/bin/env python3
"""ND-6: DASS Dropout Training.

Train WITH DASS features but randomly zero them out during training.
Forces the model to learn AV-only representations alongside AV+DASS.
At test time, DASS=0 is just another dropout event.
"""

import argparse, copy, json, math, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from ccac.baselines.anxiety_baseline import (
    BaselineConfig, _load_release_train_val, _apply_scaler, _fit_scaler,
    _build_folds, _class_weights, _classification_metrics, _set_seed,
    _resolve_device, _resolve_num_workers,
)
from ccac.baselines.dass_baseline import (
    DASSConfig, FocalLoss, DASSDataset, _extract_dass_features,
)
from ccac.experiments.deep_residual import DeepResidualModel


class DASSDropoutDataset(DASSDataset):
    """Dataset that randomly zeros DASS features at each __getitem__ call."""
    def __init__(self, *args, dass_dropout_prob=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.dass_dropout_prob = dass_dropout_prob

    def __getitem__(self, index):
        av, mask, dass, label = super().__getitem__(index)[:4]
        if self.training_enabled and torch.rand(1).item() < self.dass_dropout_prob:
            dass = torch.zeros_like(dass)
        return av, mask, dass, label


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-path', default='datasets')
    p.add_argument('--output-dir', default='artifacts/exp/nd6_dass_dropout')
    p.add_argument('--device', default='cuda')
    p.add_argument('--num-folds', type=int, default=5)
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--patience', type=int, default=12)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--focal-gamma', type=float, default=1.0)
    p.add_argument('--dass-dropout', type=float, default=0.5)
    args = p.parse_args()

    _set_seed(args.seed)
    device = _resolve_device(args.device)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'ND-6: DASS Dropout p={args.dass_dropout} | Device: {device}')

    dataset_path = Path(args.dataset_path)
    bc = BaselineConfig(
        dataset_path=str(dataset_path.resolve()),
        output_dir='.', audio_feature_name='audio_wavlm_base',
        video_feature_name='video_clip_base',
    )

    frame, av_features, clip_mask, label_mapping, input_dim = _load_release_train_val(
        bc, dataset_path)
    labels = frame['_label_index'].to_numpy(np.int64)
    dass_features = _extract_dass_features(frame, 'scores_das')
    print(f'Loaded {len(frame)} subjects, input_dim={input_dim}, dass_dim={dass_features.shape[1]}')

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

        dass_mean = dass_features[tr].mean(axis=0)
        dass_std = dass_features[tr].std(axis=0)
        dass_std = np.where(dass_std < 1e-6, 1.0, dass_std)
        train_dass = (dass_features[tr] - dass_mean) / dass_std
        val_dass = (dass_features[vl] - dass_mean) / dass_std

        # Use dropout dataset for TRAINING (drops DASS randomly)
        train_ds = DASSDataset(train_av, clip_mask[tr], train_dass, labels[tr])
        train_ds.training_enabled = True
        # Override __getitem__ with dropout
        orig_getitem = train_ds.__getitem__
        def dropout_getitem(idx, orig=orig_getitem, p=args.dass_dropout, ds=train_ds):
            items = orig(idx)
            av, mask, dass = items[0], items[1], items[2]
            if torch.rand(1).item() < p:
                dass = torch.zeros_like(dass)
            return (av, mask, dass) + items[3:]
        train_ds.__getitem__ = dropout_getitem

        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_dl = DataLoader(DASSDataset(val_av, clip_mask[vl], val_dass, labels[vl]),
                           batch_size=args.batch_size, shuffle=False, num_workers=workers)

        model = DeepResidualModel(
            input_dim=input_dim, num_classes=len(label_mapping),
            hidden_dim=256, num_heads=4, num_residual_blocks=3, dropout=0.2,
            dass_config=DASSConfig(dass_scheme='scores_das'),
        ).to(device)

        cw = _class_weights(labels[tr], len(label_mapping), 1.0)
        if cw is not None: cw = cw.to(device)
        criterion = FocalLoss(gamma=args.focal_gamma, alpha=cw)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-6)

        best_mf1, best_state, patience_ct = -math.inf, None, 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            for batch in train_dl:
                av, m, d, lb = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
                opt.zero_grad(set_to_none=True)
                loss = criterion(model(av, m, d), lb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()

            model.eval()
            vprobs_list, vlbls_list = [], []
            with torch.no_grad():
                for av, m, d, lb in val_dl:
                    av, m, lb = av.to(device), m.to(device), lb.to(device)
                    # Evaluate BOTH with and without DASS
                    logits = model(av, m, d.to(device))
                    vprobs_list.append(torch.softmax(logits, -1).cpu().numpy())
                    vlbls_list.append(lb.cpu().numpy())
            vprob = np.concatenate(vprobs_list)
            vlbl = np.concatenate(vlbls_list)
            mf1 = float(f1_score(vlbl, vprob.argmax(1), average='macro', zero_division=0))

            if mf1 > best_mf1:
                best_mf1 = mf1; patience_ct = 0
                best_state = {
                    'model': copy.deepcopy(model.state_dict()),
                    'scaler_mean': scaler[0].tolist(), 'scaler_std': scaler[1].tolist(),
                    'dass_mean': dass_mean.tolist(), 'dass_std': dass_std.tolist(),
                }
                best_probs = vprob
                # Also evaluate without DASS
                vprobs_nodass = []
                with torch.no_grad():
                    for av, m, d, lb in val_dl:
                        logits = model(av.to(device), m.to(device), torch.zeros_like(d).to(device))
                        vprobs_nodass.append(torch.softmax(logits, -1).cpu().numpy())
                vprob_nodass = np.concatenate(vprobs_nodass)
                best_probs_nodass = vprob_nodass
            else:
                patience_ct += 1
            if patience_ct >= args.patience:
                break

        model.load_state_dict(best_state['model'])
        oof_probs[vl] = best_probs_nodass  # Use no-DASS predictions for test
        oof_preds[vl] = best_probs_nodass.argmax(1)
        fm = float(f1_score(labels[vl], oof_preds[vl], average='macro', zero_division=0))
        fa = float(accuracy_score(labels[vl], oof_preds[vl]))
        print(f'MF1={fm:.4f} Acc={fa:.4f} (with DASS: {best_mf1:.4f})')
        metrics.append({'fold': fold_id, 'macro_f1': fm, 'accuracy': fa, 'mf1_with_dass': best_mf1})
        torch.save(best_state, fold_dir / 'best_model.pt')

    overall_mf1 = float(f1_score(labels, oof_preds, average='macro', zero_division=0))
    overall_acc = float(accuracy_score(labels, oof_preds))
    print(f'\nND-6 OOF (no-DASS eval): MF1={overall_mf1:.4f} Acc={overall_acc:.4f}')
    label_by_idx = {i: l for l, i in label_mapping.items()}
    print(classification_report(labels, oof_preds,
          target_names=[l for l,_ in sorted(label_mapping.items(), key=lambda x:x[1])],
          zero_division=0))

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

    summary = {'oof_mf1_nodass': overall_mf1, 'oof_acc': overall_acc, 'input_dim': input_dim,
               'dass_dropout_prob': args.dass_dropout, 'focal_gamma': args.focal_gamma,
               'label_mapping': label_mapping}
    (output_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Done. Saved to {output_dir}')


if __name__ == '__main__':
    main()
