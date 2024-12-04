# Modifications Copyright(C) [Year of 2024] Advanced Micro Devices, Inc. All rights reserved.
CUDA_VISIBLE_DEVICES="" accelerate launch train_turbo.py \
    --use_8bit_adam \
    --train_batch_size=1 \
    --n_frames 26 \
    --no_scale_pred_x0 \
    --reward_scale 1.0 \
    --video_rm_name vi_clip2 \
    --video_reward_scale 2.0 \
    --vlcd_processes  \
    --reward_train_processes  \
    --video_rm_train_processes  \
    --pretrained_model_cfg configs/inference_t2v_512_v2.0.yaml \
    --pretrained_model_path  \
    --train_shards_path_or_url  \
    --video_rm_ckpt_dir  \
    --teacher_model_path  \
    --fps 8 \
    --max_train_steps 20000 \
    --checkpoints_total_limit 10 \
    --output_dir 
 