# Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

DIMENSIONS=(
    "motion_smoothness"
)


VIDEO_FOLDER_PATHS=(
    "videos_path"
)



for VIDEO_FOLDER_PATH in "${VIDEO_FOLDER_PATHS[@]}"; do
    for DIMENSION in "${DIMENSIONS[@]}"; do
        CUDA_VISIBLE_DEVICES=6 vbench evaluate \
            --dimension ${DIMENSION} \
            --videos_path "$VIDEO_FOLDER_PATH" \
            --mode=custom_input
    done
done