# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.

#!/bin/bash

version=$1  ## 1024, 512, 256
GPU=0
crop='3-2'
ckpt='/group/ossdphi_algo_scratch_01/takisobe/i2v-turbo-v2/14000_out.ckpt'
#ckpt='331200_out.ckpt'
#ckpt='./checkpoints/dynamicrafter_512_v1/model.ckpt' 
config='configs/inference_512_v1.0_07B.yaml'
#config=configs/inference_$1_v1.0.yaml
prompt_dir="/group/ossdphi_algo_scratch_01/takisobe/VBench/crop/$crop"
#prompt_dir="prompts/512"
#base_res_dir="14000_unet_16_sp_40000_neg_fps24_$crop"
base_res_dir="14000_unet_16_sp_top10_4000_test"

# 判断分辨率
if [ "$1" == "256" ]; then
    H=256
    FS=3
elif [ "$1" == "512" ]; then
    H=320
    FS=24
elif [ "$1" == "1024" ]; then
    H=576
    FS=10
else
    echo "Invalid input. Please enter 256, 512, or 1024."
    exit 1
fi

seeds=(32324 63412 123 982 443)
#seeds=(123)

for i in {0..4}; do
    seed=${seeds[$i]}
    sub_dir=$((i + 1))  # 文件夹名称：1, 2, 3, 4, 5
    res_dir="${base_res_dir}/${sub_dir}"

    echo "Running seed=$seed -> Saving to: $res_dir"
    echo $prompt_dir

    CUDA_VISIBLE_DEVICES=$GPU python3 scripts/evaluation/inference.py \
        --seed ${seed} \
        --ckpt_path $ckpt \
        --config $config \
        --savedir $res_dir \
        --n_samples 1 \
        --bs 1 --height ${H} --width $1 \
        --unconditional_guidance_scale 7.5 \
        --ddim_steps 16 \
        --ddim_eta 1.0 \
        --prompt_dir $prompt_dir \
        --text_input \
        --video_length 16 \
        --frame_stride ${FS} \
        --use_unet 1 \
        --unet_path '/group/ossdphi_algo_scratch_01/takisobe/i2v-turbo-new_v2/output/6_6_09B_14000_bs3_topk10_stu_teacher_lr1e5_8gpu_videormbs3_mixdata_sp/checkpoint-4000/unet.pt' \
        --negative_prompt \
        --timestep_spacing 'uniform_trailing' \
        --guidance_rescale 0.7 \
        --perframe_ae
done

