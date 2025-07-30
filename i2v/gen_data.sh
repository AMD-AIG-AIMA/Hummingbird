# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.  


#!/bin/bash

# Base path variables
CSV_DIR="/scratch1_nvme_2/workspace/jieguo12/datasets-jieguo/webvid"
#CSV_DIR='/scratch1_nvme_1/workspace/takashi/OpenVid-1M/'

#CSV_DIR="/scratch/workspace/takashi/webvid/"
#VIDEO_ROOT="/scratch/workspace/takashi/webvid/train/videos"

VIDEO_ROOT="/scratch1_nvme_2/workspace/jieguo12/datasets-jieguo/webvid/train/videos"
#VIDEO_ROOT='/scratch1_nvme_1/workspace/takashi/OpenVid-1M/OpenVid-1M/video'
#SAVE_ROOT="/scratch1_nvme_2/workspace/jieguo12/datasets-jieguo/webvid/latents_no_motion_prior_16_i2v_09B_teacher"
SAVE_ROOT="/group/ossdphi_algo_scratch_01/takisobe/i2v-turbo-v2/latents_no_motion_prior_16_i2v_09B_14000"
#CKPT_PATH_teacher="/group/ossdphi_algo_scratch_01/takisobe/DynamiCrafter/checkpoints/dynamicrafter_512_v1/model.ckpt"
CKPT_PATH="14000_out.ckpt"
CKPT_PATH_teacher="/group/ossdphi_algo_scratch_01/takisobe/DynamiCrafter/checkpoints/dynamicrafter_512_v1/model.ckpt"
#PROMPT_DIR="prompts/test"

for i in {1..4}
#for i in {5..8}
do
  #GPU_ID=$((i - 1)) # GPU:0~3
  GPU_ID=$((i-1)) # GPU: 4~7
  CSV_FILE="$CSV_DIR/webvid-v2format_${i}.csv"

  CUDA_VISIBLE_DEVICES=${GPU_ID} PYTHONPATH=. \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  accelerate launch dc_gt_gen_nomotion_2.py \
  --s3_bucket_name webvid_s3_bucket \
  --path_to_csv ${CSV_FILE} \
  --raw_video_root ${VIDEO_ROOT} \
  --save_root ${SAVE_ROOT} \
  --n_frames 16 \
  --fps 24 \
  --batch_size 1 \
  --dataloader_num_workers 12 \
  --pretrained_model_path ${CKPT_PATH} \
  --pretrained_model_path_teacher ${CKPT_PATH_teacher} \
  --guidance_rescale 0.7 \
  --unconditional_guidance_scale 7.5 \
  --video_type "webvid" \
  --perframe_ae &

done


