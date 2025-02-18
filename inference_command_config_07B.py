# Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
import os
import random
device = '7'


config = "configs/inference_t2v_512_v2.0_distil_07B.yaml"
base_model = "07B_merged_all.pt"

seed = [i for i in range(1)]
prompt = '/group/ossdphi_algo_scratch_01/takisobe/VideoCrafter2/prompts/test_prompts_2.txt'
for i in seed:
    os.system(f'''
              CUDA_VISIBLE_DEVICES={device} PYTHONPATH="accelerate/t2v_turbo" python acceleration/t2v-turbo/inference_merge.py \
                --config {config}  \
                --base_model_dir {base_model} \
                --prompt {prompt} \
                --precision 16 \
                --seed {i} \
                --steps 8 \
                --frames 26 \
                --fps 8 \
                --save-path results/07B

              ''')
