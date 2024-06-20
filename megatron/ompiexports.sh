#!/bin/bash

#no need to export local rank
#let megatron initializer does this job
NUM_NODES=2
NUM_GPUS_PERNODE=2
let "NUM_GPUS = $NUM_NODES * $NUM_GPUS_PER_NODE"
export UCX_TLS=tcps
export OMPI_COMM_WORLD_SIZE=$NUM_GPUS
export OMPI_COMM_WORLD_LOCAL_SIZE=$NUM_GPUS_PER_NODE
export OMPI_UNIVERSE_SIZE=" $NUM_GPUS * 4 " #just set it to this for time being


