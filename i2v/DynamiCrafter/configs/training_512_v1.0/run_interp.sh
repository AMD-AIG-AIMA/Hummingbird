# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.

# args
name="training_512_v1.0"
config_file=configs/${name}/config_interp.yaml

# save root dir for logs, checkpoints, tensorboard record, etc.
save_root="<YOUR_SAVE_ROOT_DIR>"

mkdir -p $save_root/${name}_interp

## run
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch \
--nproc_per_node=$HOST_GPU_NUM --nnodes=1 --master_addr=127.0.0.1 --master_port=12352 --node_rank=0 \
./main/trainer.py \
--base $config_file \
--train \
--name ${name}_interp \
--logdir $save_root \
--devices $HOST_GPU_NUM \
lightning.trainer.num_nodes=1

