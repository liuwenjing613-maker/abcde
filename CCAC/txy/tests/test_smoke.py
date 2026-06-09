from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from txy.constants import CLIP_TYPES, STAGES
from txy.data.history_features import HistoryFeatureBuilder
from txy.data.longitudinal_dataset import LongitudinalPersonDataset, collate_person_batch
from txy.models.stagewise import StageWiseLongitudinalModel
from txy.constants import LEVEL_TO_INDEX
from txy.data.labels import encode_ordinal_labels
from txy.models.residual_v3 import ResidualV3Model
from txy.models.stagewise_v4 import StageWiseV4Model
from txy.training.losses import kl_distillation_loss
from txy.training.calibration import search_class_bias
from txy.training.metrics import classification_metrics


def test_history_feature_builder():
    frame = pd.DataFrame(
        {
            "t1_depression_score": [0.0, 8.0],
            "t1_anxiety_score": [0.0, 4.0],
            "t1_stress_score": [0.0, 20.0],
            "t1_depression_level": ["正常", "正常"],
            "t1_anxiety_level": ["正常", "正常"],
            "t1_stress_level": ["正常", "中度"],
            "t2_depression_score": [0.0, 12.0],
            "t2_anxiety_score": [0.0, 14.0],
            "t2_stress_score": [0.0, 18.0],
            "t2_depression_level": ["正常", "轻度"],
            "t2_anxiety_level": ["正常", "中度"],
            "t2_stress_level": ["正常", "轻度"],
            "t3_depression_score": [0.0, 10.0],
            "t3_anxiety_score": [0.0, 16.0],
            "t3_stress_score": [0.0, 14.0],
            "t3_depression_level": ["正常", "轻度"],
            "t3_anxiety_level": ["正常", "重度"],
            "t3_stress_level": ["正常", "正常"],
        }
    )
    builder = HistoryFeatureBuilder.from_labels_frame(frame)
    features, levels = builder.transform(frame)
    assert features.shape[0] == 2
    assert features.shape[1] == len(builder.feature_names)
    assert levels.shape == (2, len(builder.level_columns))


def test_stagewise_forward():
    batch_size = 2
    audio_dim, video_dim = 16, 8
    history_dim, level_slots, num_classes = 20, 9, 5
    model = StageWiseLongitudinalModel(
        audio_dim=audio_dim,
        video_dim=video_dim,
        history_score_dim=history_dim,
        history_level_slots=level_slots,
        num_classes=num_classes,
    )
    audio = torch.randn(batch_size, len(STAGES), len(CLIP_TYPES), audio_dim)
    video = torch.randn(batch_size, len(STAGES), len(CLIP_TYPES), video_dim)
    mask = torch.ones(batch_size, len(STAGES), len(CLIP_TYPES), dtype=torch.bool)
    history_scores = torch.randn(batch_size, history_dim)
    history_levels = torch.randint(0, 5, (batch_size, level_slots))
    logits = model(audio, video, mask, history_scores, history_levels)
    assert logits.shape == (batch_size, num_classes)


def test_grouped_dataset_collate():
    n = 3
    ds = LongitudinalPersonDataset(
        audio=np.zeros((n, 3, 4, 8), dtype=np.float32),
        video=np.zeros((n, 3, 4, 4), dtype=np.float32),
        fused=np.zeros((n, 3, 4, 12), dtype=np.float32),
        clip_mask=np.ones((n, 3, 4), dtype=bool),
        history_scores=np.zeros((n, 10), dtype=np.float32),
        history_levels=np.zeros((n, 9), dtype=np.int64),
        labels=np.array([0, 1, 2], dtype=np.int64),
        subject_ids=["a", "b", "c"],
    )
    batch = collate_person_batch([ds[0], ds[1]])
    assert batch.audio.shape == (2, 3, 4, 8)
    assert batch.labels.shape == (2,)


def test_ordinal_label_encoding():
    series = pd.Series(["正常", "轻度", "中度", "重度", "非常严重", "正常"])
    encoded, mapping = encode_ordinal_labels(series)
    assert mapping == LEVEL_TO_INDEX
    assert encoded.tolist() == [0, 1, 2, 3, 4, 0]


def test_residual_v3_fuse():
    batch_size = 2
    audio_dim, video_dim = 16, 8
    model = ResidualV3Model(audio_dim=audio_dim, video_dim=video_dim, num_classes=5)
    audio = torch.randn(batch_size, len(STAGES), len(CLIP_TYPES), audio_dim)
    video = torch.randn(batch_size, len(STAGES), len(CLIP_TYPES), video_dim)
    mask = torch.ones(batch_size, len(STAGES), len(CLIP_TYPES), dtype=torch.bool)
    logits_tab = torch.randn(batch_size, 5)
    history_available = torch.tensor([True, False])
    out = model(audio, video, mask, logits_tab, history_available)
    assert out["logits_final"].shape == (batch_size, 5)
    assert out["logits_mm"].shape == (batch_size, 5)
    assert torch.allclose(out["logits_final"][1], out["logits_mm"][1])


def test_stagewise_v4_forward():
    batch_size = 2
    audio_dim, video_dim = 16, 8
    model = StageWiseV4Model(audio_dim=audio_dim, video_dim=video_dim, num_classes=5)
    audio = torch.randn(batch_size, len(STAGES), len(CLIP_TYPES), audio_dim)
    video = torch.randn(batch_size, len(STAGES), len(CLIP_TYPES), video_dim)
    mask = torch.ones(batch_size, len(STAGES), len(CLIP_TYPES), dtype=torch.bool)
    out = model(audio, video, mask)
    assert out["logits_mm"].shape == (batch_size, 5)
    assert out["ordinal_logits"].shape == (batch_size, 4)
    kd = kl_distillation_loss(out["logits_mm"], out["logits_mm"].detach(), temperature=2.0)
    assert kd.ndim == 0


def test_calibration_and_metrics():
    labels = np.array([0, 1, 2, 0, 1], dtype=np.int64)
    logits = np.eye(3)[labels] * 2.0
    bias, metrics = search_class_bias(logits, labels)
    assert bias.shape == (3,)
    assert "macro_f1" in metrics
    preds = (logits + bias.reshape(1, -1)).argmax(axis=1)
    assert classification_metrics(labels, preds)["accuracy"] >= 0.0
