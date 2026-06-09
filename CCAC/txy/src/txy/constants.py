from __future__ import annotations

STAGES = ("T1", "T2", "T3")
CLIP_TYPES = ("A01", "B01", "B02", "B03")
TARGET_LABEL_COLUMN = "t4_anxiety_level"
SUBJECT_ID_COLUMNS = ("anon_school", "anon_class", "anon_person")

SCORE_COLUMNS = [
    "depression_score",
    "anxiety_score",
    "stress_score",
]
LEVEL_COLUMNS = [
    "depression_level",
    "anxiety_level",
    "stress_level",
]

# CodaBench / DASS anxiety level submission encoding:
#   0 正常      (score 0-7)
#   1 轻度      (score 8-9)
#   2 中度      (score 10-14)
#   3 重度      (score 15-19)
#   4 非常严重  (score 20+)
LEVEL_TO_INDEX = {
    "正常": 0,
    "轻度": 1,
    "中度": 2,
    "重度": 3,
    "非常严重": 4,
}

INDEX_TO_LEVEL = {index: label for label, index in LEVEL_TO_INDEX.items()}

# CodaBench submission uses DASS ordinal encoding (NOT sklearn sorted train indices).
SUBMISSION_LEVEL_TO_INDEX = LEVEL_TO_INDEX

def train_class_index_to_submission_label(class_index: int) -> int:
    """With unified ordinal encoding, train class index equals submission label."""
    code = int(class_index)
    if code not in SUBMISSION_LEVEL_TO_INDEX.values():
        raise ValueError(f"invalid class index {code}; expected 0..4 ordinal encoding")
    return code

HISTORY_SCORE_COLS = [
    f"{stage.lower()}_{col}"
    for stage in STAGES
    for col in SCORE_COLUMNS
]

HISTORY_LEVEL_COLS = [
    f"{stage.lower()}_{col}"
    for stage in STAGES
    for col in LEVEL_COLUMNS
]
