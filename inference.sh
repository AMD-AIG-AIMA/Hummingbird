# Modifications Copyright(C) [Year of 2024] Advanced Micro Devices, Inc. All rights reserved.

CUDA_VISIBLE_DEVICES=$1 PYTHONPATH="accelerate/t2v_turbo" \
python acceleration/t2v-turbo/inference_vc2_turbo.py \
--config configs/inference_t2v_512_v2.0_distil.yaml \
--base_model_dir collapsed_model.pt\
--prompt  \
--precision 16 \
--frames 26 \
--fps 8 \
--save-path results/

