# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved. 
import argparse
import logging
import os
import pickle
import random
from pathlib import Path
from einops import rearrange, repeat

# import boto3

import numpy as np
from omegaconf import OmegaConf
# import cv2
# cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False)
import torch
import torch.utils.checkpoint
from tqdm import tqdm
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

import diffusers

from data.mp4_dataset import MP4Dataset

from motion_prior_sample import get_motion_prior_score, reverse_ddim_loop
from ode_solver import DDIMSolver
from scheduler.t2v_turbo_scheduler import T2VTurboScheduler
from utils.common_utils import load_model_checkpoint
from utils.utils import instantiate_from_config
from functools import partial


logger = get_logger(__name__)

to_torch = partial(torch.tensor, dtype=torch.float32)

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    # ---------- AWS S3 Arguments ----------
    parser.add_argument(
        "--s3_bucket_name",
        type=str,
        default="BUCKET_NAME",
        help="The name of the S3 bucket.",
    )
    # ----------Model Checkpoint Loading Arguments----------
    parser.add_argument(
        "--pretrained_model_cfg",
        type=str,
        default="configs/inference_i2v_512_v2.0_distil.yaml",
        help="Pretrained Model Config.",
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="model_cache/VideoCrafter2_model.ckpt",
        help="Path to the pretrained model.",
    )

    parser.add_argument(
        "--pretrained_model_cfg_teacher",
        type=str,
        default="configs/inference_i2v_512_v2.0.yaml",
        help="Pretrained Model Config.",
    )
    parser.add_argument(
        "--pretrained_model_path_teacher",
        type=str,
        default="model_cache/VideoCrafter2_model.ckpt",
        help="Path to the pretrained model.",
    )

    # ----General Training Arguments----
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/process_no_motion",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=453645634, help="A seed for reproducible training."
    )
    # ----Logging----
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    # ----Image Processing----
    parser.add_argument(
        "--path_to_csv",
        type=str,
        default="PATH_TO_DATA_CSV",
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--raw_video_root",
        type=str,
        default="path/to/raw_video_root",
        help="The path to the root directory of the video files in a S3 bucket.",
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default="path/to/save_root",
        help="The path to the root directory of the save files in a S3 bucket.",
    )

    # ----Dataloader----
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    # ----Batch Size and Training Steps----
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=16,
        help="Number of frames to sample from a video.",
    )
    parser.add_argument(
        "--temp_loss_scale",
        type=float,
        default=100.0,
        help="Temperature scaling for the loss.",
    )
    # ----Learning Rate----
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        help="fps for the video.",
    )
    parser.add_argument(
        "--num_ddim_timesteps",
        type=int,
        default=200,
        help="Num timesteps for DDIM sampling",
    )
    parser.add_argument(
        "--max_percentage",
        type=int,
        default=0.5,
        help="Max percentage of the motion guidance percentage.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help="Negative prompt for the DDIM sampling.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="1000 (Num Train timesteps) // 50 (Num timesteps for DDIM sampling)",
    )
    # ----Distributed Training----
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
            "--guidance_rescale", type=float, default=0.0, help="guidance rescale in [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891)")
    parser.add_argument("--unconditional_guidance_scale", type=float, default=1.0, help="prompt classifier-free guidance")
    parser.add_argument("--perframe_ae", action='store_true', default=False, help="if we use per-frame AE decoding, set it to True to save GPU memory, especially for the model of 576x1024")
    parser.add_argument("--interp", action='store_true', default=False, help="generate generative frame interpolation or not")
    parser.add_argument("--height", type=int, default=320, help="image height, in pixel space")
    parser.add_argument("--width", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--prompt_dir", type=str, default=None, help="a data dir containing videos and prompts")
    parser.add_argument("--video_type", type=str, default='webvid', help="a data dir containing videos and prompts")


    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


# Adapted from pipelines.StableDiffusionPipeline.encode_prompt
def encode_prompt(prompt_batch, text_encoder, is_train=True):
    captions = []
    for caption in prompt_batch:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        prompt_embeds = text_encoder(prompt_batch)

    return prompt_embeds


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        mixed_precision="bf16" if not 'V100' in torch.cuda.get_device_name() else 'fp16',
        log_with="tensorboard",
        project_config=accelerator_project_config,
        split_batches=True,
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    device = accelerator.device
    weight_dtype = torch.bfloat16 if not 'V100' in torch.cuda.get_device_name() else torch.float16

    # 5. Load teacher Model
    config = OmegaConf.load(args.pretrained_model_cfg)
    model_config = config.pop("model", OmegaConf.create())
    model_config["params"]["unet_config"]["params"]["use_checkpoint"] = False
    pretrained_t2v = instantiate_from_config(model_config)
    pretrained_t2v = load_model_checkpoint(
        pretrained_t2v,
        args.pretrained_model_path,
    )
    pretrained_t2v.perframe_ae = args.perframe_ae

    pretrained_t2v = pretrained_t2v.to(device, weight_dtype)

    unet = pretrained_t2v.model.diffusion_model.to(device, weight_dtype)

    unet.requires_grad_(False).eval()
    pretrained_t2v.requires_grad_(False).eval()

    config = OmegaConf.load(args.pretrained_model_cfg_teacher)
    model_config = config.pop("model", OmegaConf.create())
    model_config["params"]["unet_config"]["params"]["use_checkpoint"] = False
    pretrained_t2v_teacher = instantiate_from_config(model_config)
    pretrained_t2v_teacher = load_model_checkpoint(
        pretrained_t2v_teacher,
        args.pretrained_model_path_teacher,
    )
    pretrained_t2v_teacher.perframe_ae = args.perframe_ae

    pretrained_t2v_teacher = pretrained_t2v_teacher.to(device, weight_dtype)

    unet_teacher = pretrained_t2v_teacher.model.diffusion_model.to(device, weight_dtype)

    unet_teacher.requires_grad_(False).eval()
    pretrained_t2v_teacher.requires_grad_(False).eval()


    # 1. Create the noise scheduler and the desired noise schedule.
    noise_scheduler = T2VTurboScheduler(
        linear_start=model_config["params"]["linear_start"],
        linear_end=model_config["params"]["linear_end"],
        prediction_type='v_prediction',
        rescale_betas_zero_snr=True
    )

    assert model_config["params"]["linear_start"] == 0.00085
    assert model_config["params"]["linear_end"] == 0.012

    solver = DDIMSolver(
        noise_scheduler.alphas_cumprod.numpy(),
        ddim_timesteps=args.num_ddim_timesteps,
        use_scale=True,
    ).to(device, weight_dtype)

    # Also move the alpha and sigma noise schedules to accelerator.device.
    solver = solver.to(accelerator.device)

    torch.backends.cuda.matmul.allow_tf32 = True

    dataset = MP4Dataset(
        path_to_csv=args.path_to_csv,
        video_root=args.raw_video_root,
        save_root=args.save_root,
        sample_fps=args.fps,
        sample_frames=args.n_frames,
        sample_size=list([s * 8 for s in model_config["params"]["image_size"]]),
        video_type=args.video_type,
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size * accelerator.num_processes,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    # 15. Prepare for training
    # Prepare everything with our `accelerator`.
    train_dataloader = accelerator.prepare(train_dataloader)
    #uncond_prompt_embeds = text_encoder([args.negative_prompt] * args.batch_size).to(
    #    device, weight_dtype
    #)
    #uncond_context = {"context": torch.cat([uncond_prompt_embeds], 1), "fps": args.fps}

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(
            "t2v-turbo-v2",
            config=tracker_config,
            init_kwargs={"wandb": {"name": args.output_dir.split("/")[-1]}},
        )

    # s3 = boto3.client("s3")

    progress_bar = tqdm(
        range(0, len(train_dataloader)),
        initial=0,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for batch in train_dataloader:
        # 1. Load and process the image and text conditioning
        video = batch["mp4"]
        text = batch["txt"]
        video = video.to(accelerator.device, weight_dtype, non_blocking=True)

        first_frame = video[:,0,:,:,:]

        b, t = video.shape[:2]
        pixel_values_flatten = video.view(b * t, *video.shape[2:])

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                latents = pretrained_t2v.encode_first_stage(pixel_values_flatten)
                latents = latents.view(b, t, *latents.shape[1:])
                # Convert latents from (b, t, c, h, w) to (b, c, t, h, w)
                latents = latents.permute(0, 2, 1, 3, 4)

                prompt_embeds = pretrained_t2v.get_learned_conditioning(text).to(device, weight_dtype)
                img_emb = pretrained_t2v.embedder(video[:,0])
                img_emb = pretrained_t2v.image_proj_model(img_emb)

                img_emb_teacher = pretrained_t2v_teacher.embedder(video[:,0])
                img_emb_teacher = pretrained_t2v_teacher.image_proj_model(img_emb_teacher)

                uc_prompts = args.batch_size * [""]
                uc_prompt_embs = pretrained_t2v.get_learned_conditioning(uc_prompts)
                uc_img_emb = pretrained_t2v.embedder(torch.zeros_like(video[:,0])) ## b l c
                uc_img_emb = pretrained_t2v.image_proj_model(uc_img_emb)

                uc_img_emb_teacher = pretrained_t2v_teacher.embedder(torch.zeros_like(video[:,0])) ## b l c
                uc_img_emb_teacher = pretrained_t2v_teacher.image_proj_model(uc_img_emb_teacher)


                context = {"context": torch.cat([prompt_embeds, img_emb], 1), "fs": args.fps}
                context_teacher = {"context": torch.cat([prompt_embeds, img_emb_teacher], 1), "fs": args.fps}


                uncond_context = {"context": torch.cat([uc_prompt_embs, uc_img_emb], 1), "fs": args.fps}
                uncond_context_teacher = {"context": torch.cat([uc_prompt_embs, uc_img_emb_teacher], 1), "fs": args.fps}


                img_cat_cond = latents[:,:,:1,:,:]
                img_cat_cond = repeat(img_cat_cond, 'b c t h w -> b c (repeat t) h w', repeat=latents.shape[2])

                bsz = latents.shape[0]

                # 2. Sample a random timestep for each image t_n from the ODE solver timesteps without bias.
                # For the DDIM solver, the timestep schedule is [T - 1, T - k - 1, T - 2 * k - 1, ...]
                max_inedx = int(args.num_ddim_timesteps * (1 - args.max_percentage))
                index = torch.randint(0, max_inedx, (bsz,), device=latents.device).long()
                start_timesteps = solver.ddim_timesteps[index]
                #timesteps = start_timesteps - args.topk
                #timesteps = torch.where(
                #    timesteps < 0, torch.zeros_like(timesteps), timesteps
                #)

                scale_arr1 = np.linspace(1.0, 0.7, 400)
                scale_arr2 = np.full(1000, 0.7)
                scale_arr = np.concatenate((scale_arr1, scale_arr2))
                scale_arr = to_torch(scale_arr).to(device, weight_dtype)

                latents = latents * scale_arr[start_timesteps].to(device, weight_dtype)

                # Sample noise from the prior and add it to the latents according to the noise magnitude at each
                # timestep (this is the forward diffusion process) [z_{t_{n + k}} in Algorithm 1]
                noise = torch.randn_like(latents)
                z_ts = noise_scheduler.add_noise(latents, noise, start_timesteps)
                cond_output = unet(torch.cat([z_ts,img_cat_cond], dim=1) , start_timesteps, **context)
                uncond_output = unet(torch.cat([z_ts,img_cat_cond], dim=1), start_timesteps, **uncond_context)

                cond_teacher_output = unet_teacher(torch.cat([z_ts,img_cat_cond], dim=1) , start_timesteps, **context_teacher)
                uncond_teacher_output = unet_teacher(torch.cat([z_ts,img_cat_cond], dim=1), start_timesteps, **uncond_context_teacher)


        for (
            idx,
            z_t,
            cond_out,
            uncond_out,
            cond_teacher_out,
            uncond_teacher_out,
            prompt_emb,
            relpath,
            txt,
            img_cat_cond_,
            start_timesteps_,
            img_emb_,
            uc_img_emb_,
            img_emb_teacher_,
            uc_img_emb_teacher_,
            first_frame_,
            uc_prompt_emb
        ) in zip(
            index,
            z_ts,
            cond_output,
            uncond_output,
            cond_teacher_output,
            uncond_teacher_output,
            prompt_embeds,
            batch["relpath"],
            text,
            img_cat_cond, 
            start_timesteps,
            img_emb,
            uc_img_emb,
            img_emb_teacher,
            uc_img_emb_teacher,
            first_frame,
            uc_prompt_embs,
        ):
            zeros = torch.zeros_like(z_t).to(torch.float16)
            to_save = {
                "index": idx,
                "z_t": z_t.to(torch.float16),
                "cond_out": cond_out.to(torch.float16),
                "uncond_out": uncond_out.to(torch.float16),
                "cond_teacher_out": cond_teacher_out.to(torch.float16),
                "uncond_teacher_out": uncond_teacher_out.to(torch.float16),
                "score": zeros,
                #"z_example": zeros,
                #"z_example_prev": zeros,
                "prompt_emb": prompt_emb.to(torch.float16),
                "img_cat_cond":img_cat_cond_.to(torch.float16),
                "start_timesteps":start_timesteps,
                "img_emb":img_emb_.to(torch.float16),
                "uc_prompt_emb":uc_prompt_emb.to(torch.float16),
                "uc_img_emb":uc_img_emb_.to(torch.float16),
                "img_emb_teacher":img_emb_teacher_.to(torch.float16),
                "first_frame":first_frame_.to(torch.float16),
                #"uc_img_emb_teacher":uc_img_emb_teacher_.to(torch.float16),

            }
            to_save = {k: v.detach().cpu() for k, v in to_save.items()}
            to_save["text"] = txt
            save_path = os.path.join(args.save_root, relpath.replace(".mp4", ".pkl"))
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'wb') as handle:
                pickle.dump(to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # relpath = relpath.replace(".mp4", ".pkl")
            # relpath = relpath.replace(".pkl", f"-{idx.item()}.pkl")
            # s3.put_object(
            #     Bucket=args.s3_bucket_name,
            #     Key=f"{args.save_root}/{relpath}",
            #     Body=to_save,
            # )
        progress_bar.update(1)

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
