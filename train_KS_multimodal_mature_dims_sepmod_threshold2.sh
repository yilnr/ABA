#!/bin/bash

set -u

SCRIPT="/data/zyh/NeurIPS24-LFM/train_KS_multimodal_mature_dims_sepmod_threshold.py"
BASE_CONFIG="/data/zyh/NeurIPS24-LFM/data/kinetics_sound.json"

RUN_TAG=$(date +"%Y%m%d_%H%M%S")
TMP_CFG_DIR="/data/zyh/NeurIPS24-LFM/_tmp_cfg/${RUN_TAG}"
LOG_DIR="/data/zyh/NeurIPS24-LFM/_logs/mature_sepmod/${RUN_TAG}"
SAVE_DIR="/data/zyh/NeurIPS24-LFM/_figure/mature_sepmod/${RUN_TAG}"

mkdir -p "${TMP_CFG_DIR}" "${LOG_DIR}" "${SAVE_DIR}"

# 指定三张卡
GPU0=4
GPU1=5
GPU2=6

make_cfg () {
    local in_cfg=$1
    local out_cfg=$2
    local gpu_id=$3

    python - "$in_cfg" "$out_cfg" "$gpu_id" <<'PY'
import json, sys
src, dst, gpu = sys.argv[1], sys.argv[2], sys.argv[3]
with open(src, 'r') as f:
    cfg = json.load(f)
cfg["gpu_id"] = str(gpu)
with open(dst, 'w') as f:
    json.dump(cfg, f, indent=4)
PY
}

CFG1="${TMP_CFG_DIR}/ks_gpu${GPU0}.json"
CFG2="${TMP_CFG_DIR}/ks_gpu${GPU1}.json"
CFG3="${TMP_CFG_DIR}/ks_gpu${GPU2}.json"

make_cfg "${BASE_CONFIG}" "${CFG1}" "${GPU0}"
make_cfg "${BASE_CONFIG}" "${CFG2}" "${GPU1}"
make_cfg "${BASE_CONFIG}" "${CFG3}" "${GPU2}"

EXP1_NAME="audio_q70_video_q85"
EXP2_NAME="audio_q75_video_q85"
EXP3_NAME="audio_r025_video_r035"

python "${SCRIPT}" \
    --config "${CFG1}" \
    --mature_fixed_threshold_mode warmup_quantile \
    --mature_warmup_epochs 10 \
    --audio_mature_fixed_threshold_quantile 0.70 \
    --video_mature_fixed_threshold_quantile 0.85 \
    --mature_smooth_window 3 \
    --mature_save_root "${SAVE_DIR}/${EXP1_NAME}" \
    > "${LOG_DIR}/${EXP1_NAME}.log" 2>&1 &
PID1=$!
echo "Started ${EXP1_NAME} on GPU ${GPU0}, PID=${PID1}"

python "${SCRIPT}" \
    --config "${CFG2}" \
    --mature_fixed_threshold_mode warmup_quantile \
    --mature_warmup_epochs 10 \
    --audio_mature_fixed_threshold_quantile 0.75 \
    --video_mature_fixed_threshold_quantile 0.85 \
    --mature_smooth_window 3 \
    --mature_save_root "${SAVE_DIR}/${EXP2_NAME}" \
    > "${LOG_DIR}/${EXP2_NAME}.log" 2>&1 &
PID2=$!
echo "Started ${EXP2_NAME} on GPU ${GPU1}, PID=${PID2}"

python "${SCRIPT}" \
    --config "${CFG3}" \
    --mature_fixed_threshold_mode warmup_max \
    --mature_warmup_epochs 10 \
    --audio_mature_threshold_ratio 0.25 \
    --video_mature_threshold_ratio 0.35 \
    --mature_smooth_window 3 \
    --mature_save_root "${SAVE_DIR}/${EXP3_NAME}" \
    > "${LOG_DIR}/${EXP3_NAME}.log" 2>&1 &
PID3=$!
echo "Started ${EXP3_NAME} on GPU ${GPU2}, PID=${PID3}"

wait ${PID1}
echo "${EXP1_NAME} finished."

wait ${PID2}
echo "${EXP2_NAME} finished."

wait ${PID3}
echo "${EXP3_NAME} finished."

echo "All jobs finished."
echo "Logs: ${LOG_DIR}"
echo "Figures: ${SAVE_DIR}"