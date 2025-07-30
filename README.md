<div align="center">
  <br>
  <br>
  <h1>Hummingbird: A Lightweight, High-Performance Video Generation Model</h1>
<a href='https://huggingface.co/amd/AMD-Hummingbird-T2V/tree/main'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>
<a href='https://www.amd.com/en/developer/resources/technical-articles/amd-hummingbird-0-9b-text-to-video-diffusion-model-with-4-step-inferencing.html'><img src='https://img.shields.io/badge/Technical-Blog-red'></a> 
</div>

<h2>🔆Introduction</h2>
⚡️ Hummingbird is a series of advanced video generation models developed by the AMD AIG team and trained on AMD Instinct™ MI250 GPUs. It includes text-to-video models, image-to-video models, and image/video super-resolution models. With only 0.9B parameters, the Hummingbird model demonstrates exceptional efficiency. For text-to-video tasks, it can generate text-aligned videos in just 1.87 seconds using 4 steps on an MI250 GPU. For image-to-video tasks, it takes only 11 seconds to produce high-quality 4K videos.

<div align="left">
<img src="GIFs/vbench.png" style="object-fit: contain;"/>
<em><b>Figure 1:</b> AMD Hummingbird-0.9B Visual Performance Comparison with Stat-of-the-art T2V Models on Vbench.</em>
</div>

| A cute happy Corgi playing in park, sunset, pixel.            | A cute happy Corgi playing in park, sunset, animated style.               | A cute raccoon playing guitar in the beach.                 | A cute raccoon playing guitar in the forest.                |
|------------------------|-----------------------------|-----------------------------|-----------------------------|
| <img src="GIFs/A_cute_happy_Corgi_playing_in_park,_sunset,_pixel_.gif" width="320">  | <img src="GIFs/A cute happy Corgi playing in park, sunset, animated style.gif" width="320"> | <img src="GIFs/A cute raccoon playing guitar in the beach.gif" width="320"> | <img src="GIFs/A cute raccoon playing guitar in the forest.gif" width="320"> |
|**A quiet beach at dawn and the waves gently lapping.**|**A cute teddy bear, dressed in a red silk outfit, stands in a vibrant street, chinese new year.**|**A sandcastle being eroded by the incoming tide.**|**An astronaut flying in space, in cyberpunk style.**|
|<img src="GIFs/A_quiet_beach_at_dawn_and_the_waves_gently_lapping.gif" width="320">|<img src="GIFs/A cute teddy bear, dressed in a red silk outfit, stands in a vibrant street, chinese new year..gif" width="320">|<img src="GIFs/A sandcastle being eroded by the incoming tide.gif" width="320">|<img src="GIFs/An astronaut flying in space, in cyberpunk style.gif" width="320">|
|**A cat DJ at a party.**|**A 3D model of a 1800s victorian house.**|**A drone flying over a snowy forest.**|**A ghost ship navigating through a sea under a moon.**|
|<img src="GIFs/A_cat_DJ_at_a_party.gif" width="320">|<img src="GIFs/A 3D model of a 1800s victorian house..gif" width="320">|<img src="GIFs/a_drone_flying_over_a_snowy_forest.gif" width="320">|<img src="GIFs/A_ghost_ship_navigating_through_a_sea_under_a_moon.gif" width="320">|
## 📝 Change Log
- __[2025.07.30]__: 🔥🔥Release pretrained Image-to-Video model and VSR model, and their training and inference code! 
- __[2025.03.24]__: Release [AMD-Hummingbird: Towards an Efficient Text-to-Video Model](https://arxiv.org/abs/2503.18559) Paper!
- __[2025.02.28]__: Release [Hummingbird Text-to-Video](https://www.amd.com/en/developer/resources/technical-articles/amd-hummingbird-0-9b-text-to-video-diffusion-model-with-4-step-inferencing.html) Technical Report! 
- __[2025.02.26]__: 🔥🔥Release pretrained Text-to-Video models, training and inference code! 


## 🚀Getting Started
### Installation
#### Conda
```
conda create -n AMD_Hummingbird python=3.10
conda activate AMD_Hummingbird
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/rocm6.1
pip install -r requirements.txt
```
For rocm flash-attn, you can install it by this [link](https://github.com/ROCm/flash-attention).
```
git clone https://github.com/ROCm/flash-attention.git
cd flash-attention
python setup.py install
```
It will take about 1.5 hours to install.

#### Docker
First, you should use `docker pull` to download the image.
```
docker pull rocm/vllm:rocm6.2_mi300_ubuntu20.04_py3.9_vllm_0.6.4
```
Second, you can use  `docker run` to run the image, for example:
```
docker run \
        -v "$(pwd):/workspace" \
        --device=/dev/kfd \
        --device=/dev/dri \
        -it \
        --network=host \
        --name hummingbird \
        rocm/vllm:rocm6.2_mi300_ubuntu20.04_py3.9_vllm_0.6.4
```
When you in the container, you can use `pip` to install other dependencies:
```
pip install -r requirements.txt
```
### Example Usage
#### Text-to-Video
Download the Unet pretrained checkpoint from [Hummingbird-Text-to-Video](https://huggingface.co/amd/AMD-Hummingbird-T2V/tree/main).
Run below command to generate videos:
```
# for 0.7B model
python inference_command_config_07B.py

# for 0.9B model
python inference_command_config_09B.py
```

#### Image-to-Video
Download the Image-to-Video pretrained checkpoint from [Hummingbird-Image-to-Video](https://huggingface.co/amd/AMD-Hummingbird-I2V).
Run below command to generate videos:
```
cd i2v
sh run_hummingbird.sh
```
#### Image/Video Super-Resolution
Download SR pretrained checkpoint from [Hummingbird-Text-to-Video](https://huggingface.co/amd/AMD-Hummingbird-T2V/tree/main).
Run below command to generate high-resolution videos:
```
cd VSR
sh inference_videos.sh
```
## 💥Pre-training
### Data Preparation

```
# VQA
cd data_pre_process/DOVER
sh run.sh
```
Then you can get a score table for all video qualities, sort according to the table, and remove low-scoring videos.
```
# Remove Dolly Zoom Videos
cd data_pre_process/VBench
sh run.sh 
```
According to the motion smoothness score csv file, you can  remove low-scoring videos.
### Training
#### Text-to-video

```
cd acceleration/t2v-turbo

# for 0.7 B model
sh train_07B.sh

# for 0.9 B model
sh train_09B.sh
```
#### Image/Video Super-Resolution
Firstly, you should train the Realesrnet model:
```
cd VSR
# for realesrnet model
sh train_realesrnet.sh
```
And you will get the trained checkpoint of Realesrnet, then you can train the Realesrgan model:
```
cd VSR
# for realesrgan model
sh train_realesrgan.sh
```
## 🤗Resources
### Pre-trained models
- Text-to-Video: [Hummingbird-Text-to-Video](https://huggingface.co/amd/AMD-Hummingbird-T2V/tree/main)
- Image-to-Video: [Hummingbird-Image-to-Video](https://huggingface.co/amd/AMD-Hummingbird-I2V/tree/main)
- Image/Video Super-Resolution: [Hummingbird-SR](https://huggingface.co/amd/AMD-Hummingbird-I2V/blob/main/SR.pth)
### AMD Blogs
Please refer to the following blogs to get started with using these techniques on AMD GPUs:
- [PyTorch Fully Sharded Data Parallel (FSDP) on AMD GPUs with ROCm™](https://rocm.blogs.amd.com/artificial-intelligence/fsdp-training-pytorch/README.html)
- [Accelerating Large Language Models with Flash Attention on AMD GPUs](https://rocm.blogs.amd.com/artificial-intelligence/flash-attention/README.html)
- [Accelerate PyTorch Models using torch.compile on AMD GPUs with ROCm™](https://rocm.blogs.amd.com/artificial-intelligence/torch_compile/README.html)
- [Introducing the First AMD 1B Language Models: AMD OLMo](https://www.amd.com/en/developer/resources/technical-articles/introducing-the-first-amd-1b-language-model.html)

## ❤️Acknowledgement
Our codebase builds on [VideoCrafter2](https://github.com/AILab-CVC/VideoCrafter), [DynamicCrafter](https://github.com/Doubiiu/DynamiCrafter), [T2v-Turbo](https://github.com/Ji4chenLi/t2v-turbo), [Real-ESRGAN
](https://github.com/xinntao/Real-ESRGAN).Thanks the authors for sharing their awesome codebases!

## 📋Citations
Feel free to cite our Hummingbird models and give us a star⭐, if you find our work helpful :)

```text
@article{isobe2025amd,
  title={AMD-Hummingbird: Towards an Efficient Text-to-Video Model},
  author={Isobe, Takashi and Cui, He and Zhou, Dong and Ge, Mengmeng and Li, Dong and Barsoum, Emad},
  journal={arXiv preprint arXiv:2503.18559},
  year={2025}
}
```