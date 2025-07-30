# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.

for item in *; do
    if [ -d "$item" ]; then  # Check if it's a directory
        size=$(du -sh "$item" | cut -f1)  # Human-readable size
        echo "$item: $size"
    fi
done
