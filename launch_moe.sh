#!/bin/bash

#BSUB -nnodes 2
#BSUB -W 2:00
#BSUB -P stf218
#BSUB -alloc_flags "smt4 nvme"
#BSUB -J moe
#BSUB -o logs/moe.%J.o
#BSUB -e logs/moe.%J.e
##BSUB -q debug

set +x

# load modules and conda
module load spectrum-mpi
module load open-ce/1.10.0-py311-ibm
conda activate /gpfs/alpine2/stf218/world-shared/sajal/moe-env-311

# export settings
export TORCH_EXTENSIONS_DIR=/gpfs/alpine2/stf218/world-shared/sajal/deepspeed
export HF_HOME=$PWD/../hfdata
export OMP_NUM_THREADS=1

export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=https://proxy.ccs.ornl.gov:3128/

# grab nodecount
nodes=($(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch))
nnodes=${#nodes[@]}

# launch node config
rm -f `find -name *lock`    # clear stale lock files

#python3 -m torch.distributed.run --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=<node-ip-0> -m tutel.examples.helloworld --batch_size=16
jsrun -smpiargs="-disable_gpu_hooks" -n $nnodes -r 1 -g 6 -a 6 -c 42 python3 -m torch.distributed.run --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=<node-ip-0> -m tutel.examples.helloworld --batch_size=16

jsrun --smpiargs="-disable_gpu_hooks" -n $nnodes -r 1 -g 6 -a 6 -c 42 python ./llm-classifier-ds.py \
   --model_name_or_path "/gpfs/alpine/proj-shared/stf218/llm/model/forge-m2" \
   --output_dir ./outputs \
   --do_train=True \
   --per_device_train_batch_size 1 \
   --gradient_accumulation_steps 1 \
   --max_len 512 \
   --learning_rate 0.000002 \
   --adam_beta2 0.98 \
   --weight_decay 0.0000 \
   --adam_epsilon 1e-8 \
   --num_train_epochs 20 \
   --warmup_steps 20 \
   --logging_steps 50 \
   --logging_dir "run-logs" \
   --evaluation_strategy "epoch" \
   --logging_strategy "steps" \
   --deepspeed ds_config_zero.json
#

