## Environment and packages for deepspeed-megatronLM ##

module load open-ce/1.5.2-py39-0

conda create -p path_to_condenv_dir --clone open-ce-1.5.2-py39-0

source activate path_to_condenv_dir

pip install deepspeed

git clone -b 22.12-dev https://github.com/NVIDIA/apex.git

cd apex

CXX=g++ pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./

MPICC="mpicc -shared" pip install --no-cache-dir --no-binary=mpi4py mpi4py

##
## Run smrun.sh to run deepspeed-megatron in multinode setting in olcf##
