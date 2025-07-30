# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved. 

import os
import glob
import shutil
import argparse
from tqdm import tqdm
from inference_realesrgan_video import run  # 假设主脚本为 realesrgan_inference.py
import copy


def batch_process(input_folder, output_folder, base_args):
    os.makedirs(output_folder, exist_ok=True)

    video_exts = ('.mp4', '.mov', '.avi', '.flv', '.mkv')

    video_paths = []
    for ext in video_exts:
        video_paths.extend(glob.glob(os.path.join(input_folder, f'*{ext}')))

    if not video_paths:
        print("No video files found in the input folder.")
        return

    for video_path in tqdm(video_paths, desc='Batch Video Enhancement'):
        video_name = os.path.basename(video_path)
        output_path = os.path.join(output_folder, video_name)

        args = copy.deepcopy(base_args)
        args.input = video_path
        args.output = output_folder
        # args.suffix = 'enhanced'  # 可选：如果你想加个后缀用于版本区分

        run(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch inference for all videos in a folder')
    parser.add_argument('--input_folder', type=str, required=True, help='Input folder containing videos')
    parser.add_argument('--output_folder', type=str, required=True, help='Output folder to save results')
    parser.add_argument('--model_name', type=str, default='realesr-animevideov3')
    parser.add_argument('--denoise_strength', type=float, default=0.5)
    parser.add_argument('--outscale', type=float, default=4)
    parser.add_argument('--suffix', type=str, default='enhanced')
    parser.add_argument('--tile', type=int, default=0)
    parser.add_argument('--tile_pad', type=int, default=10)
    parser.add_argument('--pre_pad', type=int, default=0)
    parser.add_argument('--face_enhance', action='store_true')
    parser.add_argument('--fp32', action='store_true')
    parser.add_argument('--fps', type=float, default=None)
    parser.add_argument('--ffmpeg_bin', type=str, default='ffmpeg')
    parser.add_argument('--extract_frame_first', action='store_true')
    parser.add_argument('--num_process_per_gpu', type=int, default=1)
    parser.add_argument('--alpha_upsampler', type=str, default='realesrgan')
    parser.add_argument('--ext', type=str, default='auto')

    args = parser.parse_args()

    batch_process(args.input_folder, args.output_folder, args)
