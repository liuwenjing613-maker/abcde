#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ccac_maps
export PYTHONPATH=src
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Device: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")')"

python scripts/exp_basic_features.py \
  --dataset-path datasets \
  --output-dir artifacts/exp/final_wider \
  --audio-feature-name audio_wavlm_base \
  --video-feature-name video_clip_base \
  --device cuda \
  --num-folds 5 \
  --hidden-dim 320 \
  --dropout 0.2 \
  --num-heads 8 \
  --num-residual-blocks 4 \
  --no-feature-cache 2>&1

echo "DONE. Check artifacts/exp/final_wider/summary.json"
