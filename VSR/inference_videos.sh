# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved. 

# Define the input and output parent directories
INPUT_PARENT="input_videos"
# Replace with your actual input parent directory
OUTPUT_PARENT="//output_sr"
# Replace with your desired output parent directory

# Model and script settings
MODEL_NAME="RealESRGAN_x4plus"
SCRIPT="inference_realesrgan_video.py"
MODEL_PATH="experiments/train_RealESRGANx4plus_32_10_16/models/net_g_500000.pth"

# Find all .mp4 video files under the input parent directory
find "$INPUT_PARENT" -type f -name "*.mp4" | while read -r video_path; do
    # Get the relative path of the video file with respect to the input parent
    rel_path="${video_path#$INPUT_PARENT/}"

    # Extract relative directory and video file name
    video_name=$(basename "$video_path")
    rel_dir=$(dirname "$rel_path")

    # Construct and create the output directory
    output_dir="$OUTPUT_PARENT/$rel_dir"
    mkdir -p "$output_dir"

    echo "Processing $rel_path..."

    # Run video super-resolution using Real-ESRGAN with custom model configuration
    CUDA_VISIBLE_DEVICES=3 python "$SCRIPT" \
        -n "$MODEL_NAME" \
        -i "$video_path" \
        -o "$output_dir" \
        --num_feat 32 \
        --num_block 10 \
        --num_grow_ch 16 \
        --model_path "$MODEL_PATH"
done
