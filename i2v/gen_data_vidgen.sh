# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.

#CSV_DIR="/scratch/workspace/takashi/vidgen"
CSV_DIR="/scratch1_nvme_2/workspace/jieguo12/datasets-jieguo/vidgen/"
#VIDEO_ROOT="/scratch/workspace/takashi/vidgen/videos"
VIDEO_ROOT="/scratch1_nvme_2/workspace/jieguo12/datasets-jieguo/vidgen/videos"

SAVE_ROOT="/group/ossdphi_algo_scratch_01/takisobe/i2v-turbo-v2/latents_no_motion_prior_16_i2v_09B_14000_vidgen"
CKPT_PATH="14000_out.ckpt"
CKPT_PATH_teacher="/group/ossdphi_algo_scratch_01/takisobe/DynamiCrafter/checkpoints/dynamicrafter_512_v1/model.ckpt"


#for i in {1..4}
for i in {5..8}
do
#  #GPU_ID=$((i - 1)) # GPU:0~3
  GPU_ID=$((i-1)) # GPU: 4~7
  CSV_FILE="$CSV_DIR/vidgen-webvidformat_${i}.csv"

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
  --video_type "vidgen" \
  --perframe_ae &

done
