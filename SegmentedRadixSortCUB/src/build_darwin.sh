#!/bin/bash
module purge
module load cuda gcc/9.3.0
#module load cgpu
#module load PrgEnv-llvm/12.0.0-git_20200824
#module switch cuda cuda/10.2.89
#module list
#set -x

rm -f *.o *.a *.so

# Pure CUDA
# Use either nvcc or clang++: both work.
# The nvcc approach requires cuda/10.2.89 because of --forward-unknown-to-host-compiler option
nvcc -c -I../cub-1.8.0 -arch=sm_70 lsort.cu -o lsort.o --forward-unknown-to-host-compiler -fPIC

#nvcc -c -I../cub-1.8.0 -arch=sm_80 -DDEBUG -DUSE_OPENMP -DLUNUS_NUM_JBLOCKS=1 -DLUNUS_NUM_IBLOCKS=1 -DDEBUG \
#     ll_lunus_proxy.cu -o ll_lunus_proxy.o --forward-unknown-to-host-compiler -fPIC
#clang++ -c -I/global/cscratch1/sd/csdaley/lunus_stuff/cub-1.8.0 -x cuda --cuda-gpu-arch=sm_70 -DLUNUS_NUM_JBLOCKS=1 -DLUNUS_NUM_IBLOCKS=1 \
#	ll_lunus_proxy.cu -o ll_lunus_proxy.o -fPIC 

g++ -c -g -O3 -fopenmp -fPIC -DLUNUS_NUM_JBLOCKS=1 -DLUNUS_NUM_IBLOCKS=1 -DDEBUG -DUSE_OPENMP -I. -o llunus_proxy.o llunus_proxy.c

#echo "Done compiling llunus_proxy.c"

# Put the CUDA code in a shared library
g++ -shared -fPIC llunus_proxy.o lsort.o -o liblunus_with_cub.so
#clang++ -shared -fPIC ll_lunus_proxy.o -o liblunus_with_cub.so

# Pure OpenMP offload.
#clang++ -g -O3 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -fPIC -DUSE_OPENMP -DUSE_OFFLOAD -DLUNUS_NUM_JBLOCKS=1 -DLUNUS_NUM_IBLOCKS=1 -DDEBUG -I. -o lunus_proxy lunus_proxy.cpp -L. -llunus_with_cub -L/usr/common/software/sles15_cgpu/cuda/10.1.243/lib64 -lcudart
gcc -g -O3 -fopenmp -fPIC -DLUNUS_NUM_JBLOCKS=1 -DLUNUS_NUM_IBLOCKS=1 -DDEBUG -I. -o lunus_proxy lunus_proxy.cpp -L. -llunus_with_cub -L${CUDA_LIB} -lcudart

file lunus_proxy
