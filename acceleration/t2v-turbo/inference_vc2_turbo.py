# Modifications Copyright(C) [Year of 2024] Advanced Micro Devices, Inc. All rights reserved.

# Adapted from https://github.com/luosiallen/latent-consistency-model
from __future__ import annotations

import argparse
import os
import random
import time
from omegaconf import OmegaConf

import gradio as gr
import numpy as np

try:
    import intel_extension_for_pytorch as ipex
except:
    pass

from utils.lora import collapse_lora, monkeypatch_remove_lora
from utils.lora_handler import LoraHandler
from utils.common_utils import load_model_checkpoint
from utils.utils import instantiate_from_config
from scheduler.t2v_turbo_scheduler import T2VTurboScheduler
from pipeline.t2v_turbo_vc2_pipeline import T2VTurboVC2Pipeline

import torch
import torchvision

from concurrent.futures import ThreadPoolExecutor
import uuid

DESCRIPTION = """# T2V-Turbo 🚀
We provide T2V-Turbo (VC2) distilled from [VideoCrafter2](https://ailab-cvc.github.io/videocrafter2/) with the reward feedback from [HPSv2.1](https://github.com/tgxs002/HPSv2/tree/master) and [InternVid2 Stage 2 Model](https://huggingface.co/OpenGVLab/InternVideo2-Stage2_1B-224p-f4).

You can download the the models from [here](https://huggingface.co/jiachenli-ucsb/T2V-Turbo-VC2). Check out our [Project page](https://t2v-turbo.github.io) 😄
"""
if torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CUDA 😀</p>"
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    DESCRIPTION += "\n<p>Running on XPU 🤓</p>"
else:
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>"
print(DESCRIPTION)

MAX_SEED = np.iinfo(np.int32).max
CACHE_EXAMPLES = torch.cuda.is_available() and os.getenv("CACHE_EXAMPLES") == "1"
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE") == "1"


"""
Operation System Options:
    If you are using MacOS, please set the following (device="mps") ;
    If you are using Linux & Windows with Nvidia GPU, please set the device="cuda";
    If you are using Linux & Windows with Intel Arc GPU, please set the device="xpu";
"""
# device = "mps"    # MacOS
# device = "xpu"    # Intel Arc GPU
device = "cuda"  # Linux & Windows


"""
   DTYPE Options:
      To reduce GPU memory you can set "DTYPE=torch.float16",
      but image quality might be compromised
"""
DTYPE = (
    torch.float16
)  # torch.float16 works as well, but pictures seem to be a bit worse


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def save_video(
    vid_tensor, metadata: dict, root_path="./", fps=16
):
    # unique_name = str(uuid.uuid4()) + ".mp4"
    unique_name = metadata['prompt'][:50].replace(' ', '_').rstrip('.')+'.mp4'
    unique_name = os.path.join(root_path, unique_name)

    video = vid_tensor.detach().cpu()
    video = torch.clamp(video.float(), -1.0, 1.0)
    video = video.permute(1, 0, 2, 3)  # t,c,h,w
    video = (video + 1.0) / 2.0
    video = (video * 255).to(torch.uint8).permute(0, 2, 3, 1)

    torchvision.io.write_video(
        unique_name, video, fps=fps, video_codec="h264", options={"crf": "10"}
    )
    return unique_name


def save_videos(
    video_array, metadata: dict, fps: int = 16, save_path='./results/'
):
    paths = []
    os.makedirs(save_path, exist_ok=True)
    with ThreadPoolExecutor() as executor:
        paths = list(
            executor.map(
                save_video,
                video_array,
                [metadata] * len(video_array),
                [save_path] * len(video_array),
                [fps] * len(video_array),
            )
        )
    return paths[0]

def generate(
    prompt: str,
    seed: int = 0,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 4,
    num_frames: int = 16,
    fps: int = 16,
    randomize_seed: bool = False,
    param_dtype="torch.float16",
    save_path = './results/'
):
    seed = randomize_seed_fn(seed, randomize_seed)
    torch.manual_seed(seed)
    pipeline.to(
        torch_device=device,
        torch_dtype=torch.float16 if param_dtype == 16 else torch.float32,
    )
    start_time = time.time()
    if param_dtype==16:
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            result = pipeline(
                prompt=prompt,
                frames=num_frames,
                fps=fps,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                num_videos_per_prompt=1,
            )
    else:
        result = pipeline(
            prompt=prompt,
            frames=num_frames,
            fps=fps,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_videos_per_prompt=1,
        )
    print(time.time() - start_time)
    paths = save_videos(
        result,
        metadata={
            "prompt": prompt,
            "seed": seed,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
        },
        fps=fps,
        save_path=save_path
    )
    print(paths)
    return paths, seed


examples = [
    "An astronaut riding a horse.",
    "Darth vader surfing in waves.",
    "Robot dancing in times square.",
    "Clown fish swimming through the coral reef.",
    "Pikachu snowboarding.",
    "With the style of van gogh, A young couple dances under the moonlight by the lake.",
    "A young woman with glasses is jogging in the park wearing a pink headband.",
    "Impressionist style, a yellow rubber duck floating on the wave on the sunset",
    "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
    "With the style of low-poly game art, A majestic, white horse gallops gracefully across a moonlit beach.",
]


if __name__ == "__main__":
    # Add model name as parameter
    parser = argparse.ArgumentParser(description="Gradio demo for T2V-Turbo.")
    parser.add_argument(
        "--base_model_dir",
        type=str,
        help="Directory of the VideoCrafter2 checkpoint.",
    )
    parser.add_argument('--config', type=str)
    parser.add_argument('--prompt', type=str)
    parser.add_argument('--randomize-seed', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--precision', type=int, default=16, choices=[16, 32])
    parser.add_argument('--guidance-scale', type=float, default=7.5)
    parser.add_argument('--steps', type=int, default=4)
    parser.add_argument('--frames', type=int, default=16)
    parser.add_argument('--fps', type=int, default=16)
    parser.add_argument('--original', action='store_true')
    parser.add_argument('--lcm', action='store_true')
    parser.add_argument('--save-path', type=str, default='./results/')


    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    model_config = config.pop("model", OmegaConf.create())
    model_config["params"]["unet_config"]["params"]["time_cond_proj_dim"] = 256
    pretrained_t2v = instantiate_from_config(model_config)
    pretrained_t2v = load_model_checkpoint(pretrained_t2v, args.base_model_dir, strict=False)

    pretrained_t2v.eval()

    scheduler = T2VTurboScheduler(
        linear_start=model_config["params"]["linear_start"],
        linear_end=model_config["params"]["linear_end"],
    )
    pipeline = T2VTurboVC2Pipeline(pretrained_t2v, scheduler, model_config)

    pipeline.to(device)
    # pipeline.to(
    #     torch_device=device,
    #     torch_dtype=torch.float16 if args.precision == 16 else torch.float32,
    # )

    if os.path.exists(args.prompt):
        with open(args.prompt, 'r') as f:
            prompts = f.readlines()
        for prompt in prompts:
            generate(
                prompt.strip(),
                args.seed,
                args.guidance_scale,
                args.steps,
                args.frames,
                args.fps,
                args.randomize_seed,
                args.precision,
                args.save_path
            )
    else:
        generate(
                args.prompt,
                args.seed,
                args.guidance_scale,
                args.steps,
                args.frames,
                args.fps,
                args.randomize_seed,
                args.precision,
                args.save_path
            )
