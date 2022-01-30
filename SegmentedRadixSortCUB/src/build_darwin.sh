#!/bin/bash
#module purge
#module load cuda gcc/9.3.0
#module load cgpu
#module load PrgEnv-llvm/12.0.0-git_20200824
#module switch cuda cuda/10.2.89
#module list
#set -x

KOKKOS_PATH=/projects/icapt/mewall/packages/kokkos-volta
KOKKOS_DEVICES=Cuda
KOKKOS_KERNELS_PATH=/projects/icapt/mewall/packages/kokkos-kernels-volta

rm -f *.o *.a *.so

# Pure CUDA
# Use either nvcc or clang++: both work.
# The nvcc approach requires cuda/10.2.89 because of --forward-unknown-to-host-compiler option
nvcc -c -I../cub-1.8.0 -arch=sm_70 -o lsort.o --forward-unknown-to-host-compiler -fPIC lsort.cu

#nvcc -c -I../cub-1.8.0 -arch=sm_80 -DDEBUG -DUSE_OPENMP -DLUNUS_NUM_JBLOCKS=1 -DLUNUS_NUM_IBLOCKS=1 -DDEBUG \
#     ll_lunus_proxy.cu -o ll_lunus_proxy.o --forward-unknown-to-host-compiler -fPIC
#clang++ -c -I/global/cscratch1/sd/csdaley/lunus_stuff/cub-1.8.0 -x cuda --cuda-gpu-arch=sm_70 -DLUNUS_NUM_JBLOCKS=1 -DLUNUS_NUM_IBLOCKS=1 \
#	ll_lunus_proxy.cu -o ll_lunus_proxy.o -fPIC 

gcc -c -g -O3 -fopenmp -fPIC -DLUNUS_NUM_JBLOCKS=1 -DLUNUS_NUM_IBLOCKS=1 -DDEBUG -DUSE_OPENMP -I. -o llunus_proxy.o llunus_proxy.c

#echo "Done compiling llunus_proxy.c"

# Put the CUDA code in a shared library
g++ -shared -fPIC llunus_proxy.o lsort.o -o liblunus_with_cub.so
#clang++ -shared -fPIC ll_lunus_proxy.o -o liblunus_with_cub.so

# Kokkos build

echo "Starting kokkos build"
#${KOKKOS_PATH}/bin/nvcc_wrapper -DDEBUG -DKOKKOS_ENABLE_CUDA -DKOKKOS_DEVICES=Cuda -DUSE_KOKKOS -fopenmp -fPIC -c -std=c++14 -I. -I/projects/icapt/mewall/packages/kokkos -I/projects/icapt/mewall/packages/kokkos/core/src -I/projects/icapt/mewall/packages/kokkos/containers/src -I/projects/icapt/mewall/packages/kokkos/algorithms/src -I/projects/icapt/mewall/packages/kokkos-kernels-volta/include -o lsort_kokkos.o lsort_kokkos.cpp
#${KOKKOS_PATH}/bin/nvcc_wrapper -arch=sm_70 -fopenmp -fPIC -DDEBUG -DUSE_KOKKOS -c -std=c++14 -I. -I${KOKKOS_PATH}/include -I/projects/icapt/mewall/packages/kokkos-kernels-volta/include -o lsort_kokkos.o lsort_kokkos.cpp
${KOKKOS_PATH}/bin/nvcc_wrapper -arch=sm_70 -fopenmp -fPIC -DDEBUG -DUSE_KOKKOS -c -std=c++14 -I. -I${KOKKOS_PATH}/include -o lsort_kokkos.o lsort_kokkos.cpp

gcc -c -O3 -fPIC -fopenmp -DLUNUS_NUM_JBLOCKS=1 -DLUNUS_NUM_IBLOCKS=1 -DDEBUG -DUSE_KOKKOS -DUSE_OPENMP -I. -o llunus_proxy_kokkos.o llunus_proxy.c

g++ -shared -fPIC llunus_proxy_kokkos.o lsort_kokkos.o -o liblunus_with_kokkos.so

# Pure OpenMP offload.
#clang++ -g -O3 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -fPIC -DUSE_OPENMP -DUSE_OFFLOAD -DLUNUS_NUM_JBLOCKS=1 -DLUNUS_NUM_IBLOCKS=1 -DDEBUG -I. -o lunus_proxy lunus_proxy.cpp -L. -llunus_with_cub -L/usr/common/software/sles15_cgpu/cuda/10.1.243/lib64 -lcudart
g++ -O3 -fopenmp -fPIC -DLUNUS_NUM_JBLOCKS=1 -DLUNUS_NUM_IBLOCKS=1 -DDEBUG -I. -o lunus_proxy lunus_proxy.cpp -L. -llunus_with_cub -L${CUDA_ROOT}/lib64 -lcudart

#g++ -O3 -fPIC -fopenmp -DLUNUS_NUM_JBLOCKS=1 -DLUNUS_NUM_IBLOCKS=1 -DDEBUG -DUSE_KOKKOS -I. -o lunus_proxy_kokkos lunus_proxy.cpp -L. -llunus_with_kokkos -L${KOKKOS_KERNELS_PATH}/lib64 -lkokkoskernels -L${KOKKOS_PATH}/lib64 -lkokkoscontainers -lkokkoscore -L${CUDA_ROOT}/lib64 -ldl -lcudart

g++ -O3 -fPIC -fopenmp -DLUNUS_NUM_JBLOCKS=1 -DLUNUS_NUM_IBLOCKS=1 -DDEBUG -DUSE_KOKKOS -I. -o lunus_proxy_kokkos lunus_proxy.cpp -L. -llunus_with_kokkos -L${KOKKOS_PATH}/lib64 -lkokkoscontainers -lkokkoscore -L${CUDA_ROOT}/lib64 -ldl -lcudart

file lunus_proxy
