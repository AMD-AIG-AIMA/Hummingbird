# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved. 
version=$1  ## 1024, 512, 256
GPU=3
ckpt='./stage_1.ckpt'
config='configs/inference_512_v1.0_09B.yaml'
prompt_dir="/VBench"
base_res_dir="result"

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


seed='123'
sub_dir='0'
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
    --unet_path 'stae_2/output/unet.pt' \
    --img_proj_path 'stage_2/output/img_proj.pt' \
    --timestep_spacing 'uniform_trailing' \
    --guidance_rescale 0.7 \
    --perframe_ae

