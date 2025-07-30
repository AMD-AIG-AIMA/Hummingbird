# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.   
import os
import difflib
import subprocess

from pathlib import Path

# Directories
dir_07B = "/group/ossdphi_algo_scratch_01/takisobe/t2v-turbo/results/07B_demo"
dir_09B = "/group/ossdphi_algo_scratch_01/takisobe/t2v-turbo-v2/results/turbo-v2/09B_16_16_no_mg"
output_dir = "./stacked_videos"
os.makedirs(output_dir, exist_ok=True)

# Load filenames
files_07B = [f for f in os.listdir(dir_07B) if f.endswith(".mp4")]
files_09B = [f for f in os.listdir(dir_09B) if f.endswith(".mp4")]

# Convert filenames (without .mp4) for fuzzy matching
names_07B = {Path(f).stem: f for f in files_07B}
names_09B = {Path(f).stem: f for f in files_09B}

# Matching loop
for name_07B_base, file_07B in names_07B.items():
    # Get the closest match from 09B (use base name for better match)
    match = difflib.get_close_matches(name_07B_base, names_09B.keys(), n=1, cutoff=0.4)

    if not match:
        print(f"[Skip] No match for: {file_07B}")
        continue

    matched_09B_base = match[0]
    file_09B = names_09B[matched_09B_base]

    path_07B = os.path.join(dir_07B, file_07B)
    path_09B = os.path.join(dir_09B, file_09B)
    output_path = os.path.join(output_dir, f"{matched_09B_base}_stacked.mp4")

    print(f"[Match] 07B: {file_07B}  <--> 09B: {file_09B}")

    # FFmpeg command to stack the videos side-by-side
    cmd = [
        "ffmpeg",
        "-i", path_07B,
        "-i", path_09B,
        "-filter_complex",
        "[0:v]scale=iw:ih,setsar=1[v0];[1:v]scale=iw:ih,setsar=1[v1];"
        "[v0][v1]hstack=shortest=1[outv]",
        "-map", "[outv]",
        "-y", output_path
    ]
    subprocess.run(cmd)

print("✅ All possible video pairs stacked.")
