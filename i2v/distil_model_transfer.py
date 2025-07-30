# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.  
import torch
import os

# Load the model checkpoint
#checkpoint_path = "small_cvpr_second.ckpt"  # Replace with your checkpoint path
checkpoint_path = "97500.ckpt"
checkpoint = torch.load(checkpoint_path,map_location='cpu')
#checkpoint = checkpoint.cuda()

# Get the model's state_dict
state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

# Filter out layers with 'teacher' in their name
filtered_state_dict = {k: v for k, v in state_dict.items() if 'teacher_model' not in k}
#print(filtered_state_dict)

# Update the checkpoint with the filtered state_dict
checkpoint['state_dict'] = filtered_state_dict

# Save the modified checkpoint
base, ext = os.path.splitext(checkpoint_path)
new_checkpoint_path = base + '_out' + ext  # Replace with desired path
torch.save(checkpoint, new_checkpoint_path)

print(f"Checkpoint saved to {new_checkpoint_path}")
