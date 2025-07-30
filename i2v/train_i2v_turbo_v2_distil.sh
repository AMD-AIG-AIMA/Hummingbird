# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.

#--pretrained_model_path_teacher /group/ossdphi_algo_scratch_01/takisobe/DynamiCrafter/checkpoints/dynamicrafter_512_v1/model.ckpt \
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' \
accelerate launch --main_process_port 29789 train_latent_i2v_turbo_v2.py \
    --pretrained_model_cfg_teacher configs/inference_i2v_512_v2.0.yaml \
    --pretrained_model_path_teacher /group/ossdphi_algo_scratch_01/takisobe/DynamiCrafter/checkpoints/dynamicrafter_512_v1/model.ckpt \
    --pretrained_model_cfg_stu configs/inference_i2v_512_v2.0_distil.yaml \
    --pretrained_model_path_stu 14000_out.ckpt \
    --output_dir output/6_6_09B_14000_bs3_topk10_stu_teacher_lr1e5_8gpu_videormbs3_mixdata_sp_imgemb \
    --train_shards_path_or_url /group/ossdphi_algo_scratch_01/takisobe/i2v-turbo-v2/latents_final_14000_vidgen_webvid.csv \
    --latent_root_webvid /group/ossdphi_algo_scratch_01/takisobe/i2v-turbo-v2/latents_no_motion_prior_16_i2v_09B_14000 \
    --latent_root_vidgen /group/ossdphi_algo_scratch_01/takisobe/i2v-turbo-v2/latents_no_motion_prior_16_i2v_09B_14000_vidgen \
    --gpu_type 'nvidia' \
    --mixed_precision bf16 \
    --allow_tf32 \
    --n_frames 16 \
    --reward_scale 0.2 \
    --video_reward_scale 0.5 \
    --fps 8 \
    --train_batch_size 3 \
    --percentage 0.5 \
    --motion_gs 0.0 \
    --reward_train_bsz 1 \
    --reward_frame_bsz 2 \
    --video_rm_frame_bsz 4 \
    --reward_fn_name weighted_hpsv2_clip \
    --video_rm_name vi_clip2 \
    --ddim_eta 0.0 \
    --train_img_emb \
    --video_rm_ckpt_dir model_cache/InternVideo2-stage2_1b-224p-f4.pt
