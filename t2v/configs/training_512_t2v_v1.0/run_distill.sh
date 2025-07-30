# Modifications Copyright(C) [Year of modification] Advanced Micro Devices, Inc. All rights reserved

# NCCL configuration
# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=3
# export NCCL_NET_GDR_LEVEL=3
# export NCCL_TOPO_FILE=/tmp/topo.txt

CUDA_DEVICES='0,1,2,3,4,5,6,7'
# args
name="training_512_t2v_v1.0"
config_file=configs/${name}/config_distil.yaml # change for your config

# save root dir for logs, checkpoints, tensorboard record, etc.
save_root="./t2v_distil"


mkdir -p $save_root/$name
export HOST_GPU_NUM=8

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES torchrun --nproc_per_node $HOST_GPU_NUM --master_port 49555 \
./main/distiller.py \
--base $config_file \
--train \
--name $name \
--logdir $save_root \
--devices $HOST_GPU_NUM \
lightning.trainer.num_nodes=1