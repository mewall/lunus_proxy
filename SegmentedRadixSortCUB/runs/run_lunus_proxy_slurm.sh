#!/bin/bash
#SBATCH --job-name=lunus    # Job name
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH -t 00:40:00
#SBATCH -c 10
#SBATCH -G 4
#SBATCH -A m1759
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --output=lunus.log
# #SBATCH --exclusive

#module purge
#module load cgpu
#module load PrgEnv-llvm/12.0.0-git_20200824
#module switch cuda cuda/10.2.89

BASE_PATH=/projects/lunus/exafel/lunus_proxy/SegmentedRadixSortCUB

LUNUS_PATH=${BASE_PATH}/src
IMAGE_PATH=${BASE_PATH}/data

export LD_LIBRARY_PATH=${LUNUS_PATH}:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${CUDA_LIB}:$LD_LIBRARY_PATH
echo '*************************************'
pwd; hostname; date
echo '*************************************'

export OMP_NUM_THREADS=8

# export LIBOMPTARGET_DEBUG=4
#nsys nvprof ${LUNUS_PATH}/lunus_proxy ${IMAGE_PATH}/snc_newhead_00001.img out.img
${LUNUS_PATH}/lunus_proxy ${IMAGE_PATH}/snc_newhead_00001.img out.img

echo '*************************************'
date
echo '*************************************'
