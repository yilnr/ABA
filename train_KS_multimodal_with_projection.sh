
python /data/zyh/NeurIPS24-LFM/train_KS_multimodal_with_projection.py \
  --config /data/zyh/NeurIPS24-LFM/data/kinetics_sound.json \
  --proj_weight 0.2 \
  --proj_momentum 0.9 \
  --proj_eps 1e-5 \
  --proj_loss smooth_l1 \
  --proj_cosine_weight 0.1