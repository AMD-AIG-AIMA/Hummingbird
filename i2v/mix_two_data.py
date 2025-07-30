# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.   
import pandas as pd


vidgen_path = 'latents_final_14000_vidgen.csv'
webvid_path = 'latents_final_14000.csv'


vidgen_df = pd.read_csv(vidgen_path)
webvid_df = pd.read_csv(webvid_path)


vidgen_sample = vidgen_df.sample(frac=1, random_state=42)
webvid_sample = webvid_df.sample(frac=1, random_state=42)


vidgen_sample['data_name'] = 'vidgen'
webvid_sample['data_name'] = 'webvid'


combined_df = pd.concat([vidgen_sample, webvid_sample], ignore_index=True)
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)  # 打乱顺序


combined_df = combined_df[['relpath', 'text', 'data_name']]


output_path = 'latents_final_14000_vidgen_webvid.csv'
combined_df.to_csv(output_path, index=False)

print(f"✅ 合并完成，文件已保存至: {output_path}")
