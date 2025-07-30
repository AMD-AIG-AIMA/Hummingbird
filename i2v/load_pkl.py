# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
import pickle

with open('/scratch1_nvme_2/workspace/jieguo12/datasets/webvid/latents_with_motion_prior/007051_007100/22052140.pkl', 'rb') as f:
    data = pickle.load(f)

print("Keys in the loaded data:")
#print(data.keys())
print(data)

