#!/bin/bash

NUM_NODES=2
NUM_GPUS_PERNODE=2
CPU_PER_TASK=7
SMT_LVL=4  #default value. change if smt alloc_flag is set

let "NUM_THREADS = $SMT_LVL * $CPU_PER_TASK" 
let "RES_SET = $NUM_NODES * $NUM_GPUS_PERNODE"

export CXX=g++ #uncomment it if torch script needs to compile
export UCX_TLS=tcps
export OMPI_COMM_WORLD_SIZE=$RES_SET
export OMPI_COMM_WORLD_LOCAL_SIZE=$NUM_GPUS_PER_NODE
export WORLD_SIZE=$OMPI_COMM_WORLD_SIZE # manually set for megatron
export OMPI_UNIVERSE_SIZE=" $RES_SET * 4 " #not important unless more mpi tasks needed to be spwaned

script="./gptmoeds.sh $NUM_NODES $NUM_GPUS_PERNODE $RES_SET"

jsrun -n$RES_SET -r$NUM_GPUS_PERNODE -c$CPU_PER_TASK -a1 -g1  -b packed:$CPU_PER_TASK -d packed -EOMP_NUM_THREADS=$NUM_THREADS $script

