#!/bin/bash

# Define an array of video folder paths
VIDEO_FOLDERS=(
    "/group/ossdphi_algo_scratch_01/takisobe/VideoCrafter2/results_ori/base_512_v2"
    "/group/ossdphi_algo_scratch_01/takisobe/VideoCrafter2/results_ori_26/base_512_v2"
    "/group/ossdphi_algo_scratch_01/takisobe/jun_vsr/test_lenovo/test_ori_sr"
    "/group/ossdphi_algo_scratch_01/takisobe/VideoCrafter2/test_130000_24_8/base_512_v2"
    "/group/ossdphi_algo_scratch_01/takisobe/jun_vsr/test_lenovo/test_good_sr"
    "/group/ossdphi_algo_scratch_01/takisobe/VideoCrafter2/test_140000_26_8/base_512_v2"
    "/group/ossdphi_algo_scratch_01/takisobe/VideoCrafter2/test_140000_35_8/base_512_v2"
    "/group/ossdphi_algo_scratch_01/takisobe/VideoCrafter2/test_140000_45_8/base_512_v2"
    "/group/ossdphi_algo_scratch_01/takisobe/VideoCrafter2/test_140000_50_8/base_512_v2"

)

# Loop through each video folder and call the Python script
for folder in "${VIDEO_FOLDERS[@]}"
do
    echo "Processing $folder"
    python cal_optical.py "$folder"
done