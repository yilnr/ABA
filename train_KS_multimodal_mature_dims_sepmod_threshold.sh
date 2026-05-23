#!/bin/bash

set -u

# =========================
# 基本路径
# =========================
SCRIPT="/data/zyh/NeurIPS24-LFM/train_KS_multimodal_mature_dims_sepmod_threshold.py"
CONFIG="/data/zyh/NeurIPS24-LFM/data/kinetics_sound.json"

RUN_TAG=$(date +"%Y%m%d_%H%M%S")
LOG_ROOT="/data/zyh/NeurIPS24-LFM/_logs/mature_sepmod/${RUN_TAG}"
SAVE_ROOT="/data/zyh/NeurIPS24-LFM/_figure/mature_sepmod/${RUN_TAG}"

mkdir -p "${LOG_ROOT}"
mkdir -p "${SAVE_ROOT}"

echo "RUN_TAG=${RUN_TAG}"
echo "LOG_ROOT=${LOG_ROOT}"
echo "SAVE_ROOT=${SAVE_ROOT}"

# =========================
# 显卡分配
# 改这里
# =========================
GPU0=4
GPU1=5
GPU2=5

# =========================
# 实验 1
# =========================
EXP1_NAME="audio_q70_video_q85"
python "${SCRIPT}" \
    --config "${CONFIG}" \
    --gpu_id "${GPU0}" \
    --mature_fixed_threshold_mode warmup_quantile \
    --mature_warmup_epochs 10 \
    --audio_mature_fixed_threshold_quantile 0.60 \
    --video_mature_fixed_threshold_quantile 0.75 \
    --mature_smooth_window 3 \
    --mature_save_root "${SAVE_ROOT}/${EXP1_NAME}" \
    > "${LOG_ROOT}/${EXP1_NAME}.log" 2>&1 &

PID1=$!
echo "Started ${EXP1_NAME} on GPU ${GPU0}, PID=${PID1}"

# =========================
# 实验 2
# =========================
EXP2_NAME="audio_q75_video_q85"
python "${SCRIPT}" \
    --config "${CONFIG}" \
    --gpu_id "${GPU1}" \
    --mature_fixed_threshold_mode warmup_quantile \
    --mature_warmup_epochs 10 \
    --audio_mature_fixed_threshold_quantile 0.55 \
    --video_mature_fixed_threshold_quantile 0.65 \
    --mature_smooth_window 3 \
    --mature_save_root "${SAVE_ROOT}/${EXP2_NAME}" \
    > "${LOG_ROOT}/${EXP2_NAME}.log" 2>&1 &

PID2=$!
echo "Started ${EXP2_NAME} on GPU ${GPU1}, PID=${PID2}"

# =========================
# 实验 3
# =========================
EXP3_NAME="audio_r025_video_r035"
python "${SCRIPT}" \
    --config "${CONFIG}" \
    --gpu_id "${GPU2}" \
    --mature_fixed_threshold_mode warmup_max \
    --mature_warmup_epochs 10 \
    --audio_mature_threshold_ratio 0.50 \
    --video_mature_threshold_ratio 0.60 \
    --mature_smooth_window 3 \
    --mature_save_root "${SAVE_ROOT}/${EXP3_NAME}" \
    > "${LOG_ROOT}/${EXP3_NAME}.log" 2>&1 &

PID3=$!
echo "Started ${EXP3_NAME} on GPU ${GPU2}, PID=${PID3}"

# =========================
# 等待
# =========================
wait ${PID1}
echo "${EXP1_NAME} finished."

wait ${PID2}
echo "${EXP2_NAME} finished."

wait ${PID3}
echo "${EXP3_NAME} finished."

echo "All jobs finished."