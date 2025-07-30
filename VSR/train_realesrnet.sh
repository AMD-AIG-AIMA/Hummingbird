# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.  

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    torchrun --nproc_per_node=8 \
    --master_port=4321 realesrgan/train.py \
    -opt options/train_realesrnet_x4plus.yml \
    --launcher pytorch \
    --auto_resume