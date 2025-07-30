# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import os
import shutil
import json
import argparse

def find_all_matching_prompts(filename_base, prompt_list):
    filename_base = filename_base.lower().strip()
    return [prompt for prompt in prompt_list if filename_base in prompt.lower()]

def main(video_dir, json_file):
    input_root = os.path.normpath(video_dir)
    output_root = input_root + "_vbench"

    if not os.path.exists(input_root):
        raise FileNotFoundError(f"❌ input_root '{input_root}' does not exist.")

    folder_suffix_map = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4}
    missing = [f for f in folder_suffix_map if not os.path.isdir(os.path.join(input_root, f))]
    if missing:
        raise FileNotFoundError(f"❌ Missing subfolders: {missing}")

    count_map = {}
    for f in folder_suffix_map:
        files = [x for x in os.listdir(os.path.join(input_root, f)) if x.endswith('.jpg.mp4')]
        count_map[f] = len(files)
    if len(set(count_map.values())) != 1:
        raise ValueError(f"❌ Inconsistent video counts: {count_map}")
    else:
        print(f"✅ All folders have {list(count_map.values())[0]} videos.")

    with open(json_file, 'r') as f:
        metadata = json.load(f)
    prompt_list = [entry["prompt_en"] for entry in metadata]

    os.makedirs(output_root, exist_ok=True)

    for folder_name, suffix in folder_suffix_map.items():
        folder_path = os.path.join(input_root, folder_name)
        for file in os.listdir(folder_path):
            if not file.endswith(".jpg.mp4"):
                continue

            filename_base = file.replace(".jpg.mp4", "").lower().strip()
            matched_prompts = find_all_matching_prompts(filename_base, prompt_list)

            if not matched_prompts:
                print(f"⚠️  No match found for {file}, skipping.")
                continue

            src_path = os.path.join(folder_path, file)
            for prompt in matched_prompts:
                new_name = f"{prompt}-{suffix}.mp4"
                dst_path = os.path.join(output_root, new_name)
                shutil.copy2(src_path, dst_path)

    # Final video count
    saved_videos = [f for f in os.listdir(output_root) if f.endswith('.mp4')]
    print(f"📦 Total saved videos in '{output_root}': {len(saved_videos)}")
    print("✅ Done: All videos processed and renamed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename videos using all matching prompt_en substrings.")
    parser.add_argument("--video", type=str, required=True, help="Input folder with subfolders 1~5")
    args = parser.parse_args()

    json_file = 'vbench2_i2v_full_info.json'
    main(args.video, json_file)

