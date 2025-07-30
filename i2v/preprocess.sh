# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.

CUDA_VISIBLE_DEVICES='0,1,2,3' PYTHONPATH=. \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
accelerate launch preprocess_scripts/preprocess_with_motion_prior.py \
--s3_bucket_name webvid_s3_bucket \
--path_to_csv /scratch1_nvme_2/workspace/jieguo12/datasets/webvid/webvid-train-partial.csv \
--raw_video_root /scratch1_nvme_2/workspace/jieguo12/datasets/webvid/train \
--save_root /scratch1_nvme_2/workspace/jieguo12/datasets/webvid/train/latents_with_motion_26 \
--n_frames 26 \
--fps 8 \
--batch_size 1 \
--dataloader_num_workers 12 \
--pretrained_model_path /group/ossdphi_algo_scratch_01/takisobe/VideoCrafter2/checkpoints/base_512_v2/ori_model.ckpt
