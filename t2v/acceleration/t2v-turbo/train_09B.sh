# Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --main_process_port 1997 train_turbo.py \
    --use_8bit_adam \
    --train_batch_size=1 \
    --n_frames 26 \
    --no_scale_pred_x0 \
    --reward_scale 1.0 \
    --video_rm_name vi_clip2 \
    --video_reward_scale 2.0 \
    --vlcd_processes "0,1,2" \
    --reward_train_processes "0,1,2" \
    --video_rm_train_processes "3," \
    --pretrained_model_cfg configs/inference_t2v_512_v2.0_distil_09B.yaml \
    --pretrained_model_path small_cvpr_second_09B_58000_12000_out.ckpt \
    --train_shards_path_or_url webvid-train-partial.csv \
    --video_rm_ckpt_dir models--OpenGVLab--InternVideo2-Stage2_1B-224p-f4/snapshots/4362e1f88a992e7edbfd7696f7f78b7f79426dfd/InternVideo2-stage2_1b-224p-f4.pt \
    --teacher_model_path VideoCrafter/checkpoints/base_512_v2/model.ckpt \
    --fps 8 \
    --max_train_steps 400000 \
    --output_dir output/09B \
    --checkpointing_steps 1000 \
    --checkpoints_total_limit 20 \
    --seed 123456

