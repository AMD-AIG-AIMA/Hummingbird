# download from https://github.com/Ji4chenLi/t2v-turbo/tree/main/data

import pickle
import random

import pandas as pd
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset

# import boto3
from diffusers.utils import logging
import sys
import os

from utils.common_utils import read_video_to_tensor

logger = logging.get_logger(__name__)


class MP4Dataset(Dataset):
    def __init__(
        self,
        path_to_csv,
        video_root="video_root",
        save_root='latents',
        sample_fps=8,
        sample_frames=16,
        sample_size=[320, 512],
        bucket="BUCKET_NAME",
        data_type='webvid'
    ):
        self.video_root = video_root
        self.save_root = save_root
        # self.s3_client = boto3.client("s3")
        # self.bucket = bucket

        logger.info(f"loading annotations from {path_to_csv} ...")
        self.video_df = pd.read_csv(path_to_csv)
        self.length = len(self.video_df)
        logger.info(f"data scale: {self.length}")

        self.sample_fps = sample_fps
        self.sample_frames = sample_frames
        self.data_type = data_type

        sample_size = (
            tuple(sample_size)
            if not isinstance(sample_size, int)
            else (sample_size, sample_size)
        )
        self.pixel_transforms = transforms.Compose(
            [
                transforms.Resize(sample_size),
                transforms.CenterCrop(sample_size),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
                ),
            ]
        )

    def get_video_text_pair(self, idx):
        video_dict = self.video_df.iloc[idx].to_dict()
        relpath, text = video_dict["relpath"], video_dict["text"]
        if os.path.exists(os.path.join(self.save_root, relpath)):
            raise Exception("pickle already exists")
        if self.data_type == 'webvid':
            video_dir = f"{self.video_root}/{relpath.replace('.pkl', '.mp4')}"
        else:
            video_dir = f"{self.video_root}/{relpath}"
            print(video_dir)

        # data_body = self.s3_client.get_object(Bucket=self.bucket, Key=video_dir).get(
        #     "Body"
        # )
        pixel_values = read_video_to_tensor(
            video_dir,
            self.sample_fps,
            self.sample_frames,
            uniform_sampling=False,
        )
        return pixel_values, text, relpath

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, text, relpath = self.get_video_text_pair(idx)
                break
            except Exception as e:
                # logger.info(idx, e)
                idx = random.randint(0, self.length - 1)

        pixel_values = self.pixel_transforms(pixel_values)
        sample = dict(mp4=pixel_values, txt=text, relpath=relpath)
        return sample


class MP4LatentDataset(Dataset):
    def __init__(
        self,
        path_to_csv,
        latent_root_webvid="latent_root",
        latent_root_vidgen="latent_root",
        bucket="BUCKET_NAME",
    ):
        # self.s3_resource = boto3.resource("s3")
        # self.bucket = bucket
        self.latent_root_vidgen = latent_root_vidgen
        self.latent_root_webvid = latent_root_webvid

        logger.info(f"loading annotations from {path_to_csv} ...")
        self.latent_df = pd.read_csv(path_to_csv)
        self.length = len(self.latent_df)
        logger.info(f"data scale: {self.length}")

    def get_latent_text_pair(self, idx):
        latent_dict = self.latent_df.iloc[idx].to_dict()
        relpath, text, data_name = latent_dict["relpath"], latent_dict["text"], latent_dict["data_name"]

        if data_name == 'webvid':
            if latent_dict.get("latent_root", None) is not None:
                latent_dir = f"{latent_dict['latent_root']}/{relpath}"
            else:
                latent_dir = f"{self.latent_root_webvid}/{relpath}"

        elif data_name == 'vidgen':
            if latent_dict.get("latent_root", None) is not None:
                latent_dir = f"{latent_dict['latent_root']}/{relpath}"
            else:
                latent_dir = f"{self.latent_root_vidgen}/{relpath}"

        if "use_motion_guide" in latent_dict:
            use_motion_guide = bool(latent_dict["use_motion_guide"])
        else:
            use_motion_guide = True

        if "short_text" in latent_dict:
            short_text = latent_dict["short_text"]
        else:
            short_text = text

        if str(short_text) == "nan":
            short_text = ""

        # latent_dict = pickle.loads(
        #     self.s3_resource.Bucket(self.bucket).Object(latent_dir).get()["Body"].read()
        # )

        with open(latent_dir, 'rb') as handle:
            latent_dict = pickle.load(handle)

        if "webvid" in latent_dir:
            text = latent_dict.pop("text")
            short_text = text
        elif "text" in latent_dict:
            assert text == latent_dict.pop("text")
        return latent_dict, text, short_text, use_motion_guide

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # NOTE: Remove the while loop if you are debugging the dataset
        while True:
            try:
                latent_dict, text, short_text, use_motion_guide = (
                    self.get_latent_text_pair(idx)
                )
                for k in latent_dict.keys():
                    if isinstance(latent_dict[k], torch.Tensor):
                        latent_dict[k] = latent_dict[k].detach().cpu()
                sample = dict(
                    txt=text, short_txt=short_text, use_motion_guide=use_motion_guide
                )
                if 'z_example' not in latent_dict:
                    latent_dict['z_example'] = torch.zeros((4, 16, 40, 64))

                if 'z_example_prev' not in latent_dict:
                    latent_dict['z_example_prev'] = torch.zeros((4, 16, 40, 64))

                if 'uc_img_emb' not in latent_dict:
                    latent_dict['uc_img_emb'] = torch.zeros((256,1024))

                if 'uc_img_emb_teacher' not in latent_dict:
                    latent_dict['uc_img_emb_teacher'] = torch.zeros((256,1024))

                if 'first_frame' not in latent_dict:
                    latent_dict['first_frame'] = torch.zeros((3,320,512))
                sample.update(latent_dict)
                break
            except Exception as e:
                logger.info(idx, e)
                idx = random.randint(0, self.length - 1)
        return sample


if __name__ == "__main__":
    import torchvision
    from torch.utils.data import DataLoader

    # random_indx = list(range(10))
    # dataset = MP4LatentDataset("data/mixed_motion_latent_128k_webvid.csv")
    # data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    dataset = MP4Dataset(
        path_to_csv='/group/ossdphi_algo_scratch_08/jieguo12/datasets/vidgen/vidgen-part.csv',
        video_root='/group/ossdphi_algo_scratch_08/jieguo12/datasets/vidgen/videos',
        save_root='/group/ossdphi_algo_scratch_08/jieguo12/datasets/vidgen/latents',
        sample_fps=8,
        sample_frames=26,
        sample_size=list([320, 512]),
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    for i, sample in enumerate(data_loader):
        # print(sample["txt"])
        # print(sample["short_txt"])
        # print(sample["use_motion_guide"])
        # print(sample["index"])
        # for k, v in sample.items():
        #     if isinstance(v, torch.Tensor):
        #         print(k, v.shape)
        # break

