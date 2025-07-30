# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved. 
import os
import pandas as pd

# Load the filtered CSV
df = pd.read_csv('latents_filtered_14000.csv')
#df = pd.read_csv('latents_filtered_14000_vidgen.csv')

# Create the 'relpath' column
# webvid
df['relpath'] = df.apply(lambda row: os.path.join(f"{row['relpath']}"), axis=1)

# vidgen
#df['relpath'] = df.apply(lambda row: os.path.join(f"{row['videoid']}.pkl"), axis=1)



# Rename 'name' column to 'text'
df = df.rename(columns={'name': 'text'})

# Keep only 'relpath' and 'text' columns
final_df = df[['relpath', 'text']]

# Save to new CSV
final_df.to_csv('latents_final_14000.csv', index=False)

#final_df.to_csv('latents_final_14000_vidgen.csv', index=False)


print(f"Saved new CSV with paths as 'latents_final.csv'.")
