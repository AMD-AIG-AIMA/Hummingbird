# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import argparse
import cv2
import glob
import os


def main(args):
    txt_file = open(args.meta_info, 'w')

    # recursively find all image files under the input root
    img_paths = sorted(glob.glob(os.path.join(args.input, '*', '*')))  # e.g. A/*/*
    for img_path in img_paths:
        status = True
        if args.check:
            try:
                img = cv2.imread(img_path)
            except (IOError, OSError) as error:
                print(f'Read {img_path} error: {error}')
                status = False
            if img is None:
                print(f'Img is None: {img_path}')
                status = False
        if status:
            img_rel_path = os.path.relpath(img_path, args.input)
            print(img_rel_path)
            txt_file.write(f'{img_rel_path}\n')


if __name__ == '__main__':
    """
    Generate meta info for all images under each subfolder of a given folder A.
    Each line in the output txt records a relative path like: subfolder1/image001.png
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                        help='Root folder (A) that contains subfolders with images')
    parser.add_argument('--meta_info', type=str, default='meta_info.txt',
                        help='Path to save the output txt file')
    parser.add_argument('--check', action='store_true',
                        help='Enable reading image to check for errors')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.meta_info), exist_ok=True)

    main(args)
