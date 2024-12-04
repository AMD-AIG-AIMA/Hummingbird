# Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.


SPECIFIED_DIR="Videos_path"
current_path=$(pwd)
OUTPUT_CSV_PATH="${current_path}/results/results.csv"
CUDA_VISIBLE_DEVICES=5 python evaluate_a_set_of_videos.py -in ${SPECIFIED_DIR} -out ${OUTPUT_CSV_PATH}
