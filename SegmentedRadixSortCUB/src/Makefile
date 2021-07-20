# module purge
# module load cgpu
# module load PrgEnv-llvm/12.0.0-git_20200824
# module switch cuda cuda/10.2.89
# module list

NVCC = nvcc
CC = clang++

# CFLAGS = -g -O3 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -fPIC -DUSE_OPENMP -DUSE_OFFLOAD -DLUNUS_NUM_JBLOCKS=1 -DLUNUS_NUM_IBLOCKS=1 -DDEBUG -I.
CFLAGS = -g -O3 -fPIC -DLUNUS_NUM_JBLOCKS=1 -DLUNUS_NUM_IBLOCKS=1 -DDEBUG -I.
LFLAGS = -llunus_proxy_shared
NVCCFLAGS = -c 
CUB_PATH=/global/cscratch1/sd/pcarrier/ExaFEL/SegmentedRadixSortCUB/cub-1.8.0
# Note that the CUDA PATH is not the same as the module number, which explains the use of the --forward-unknown-to-host-compiler
CUDA_PATH=/usr/common/software/sles15_cgpu/cuda/10.1.243

all: build 

# Pure OpenMP offload. Main program
build: ll_lunus_code archive_it
	$(CC) $(CFLAGS) -o lunus_proxy lunus_proxy.cpp -L. -llunus_with_cub -L$(CUDA_PATH)/lib64 -lcudart

# The nvcc approach requires cuda/10.2.89 because of --forward-unknown-to-host-compiler option
# -fopenmp -DUSE_OPENMP 
ll_lunus_code:
	$(NVCC) $(NVCCFLAGS) -I$(CUB_PATH) -arch=sm_70 -DLUNUS_NUM_JBLOCKS=1 -DLUNUS_NUM_IBLOCKS=1 -DDEBUG ll_lunus_proxy.cu -o ll_lunus_proxy.o --forward-unknown-to-host-compiler -fPIC  -lcudart

# Put the CUDA code in a shared library:
archive_it:
	clang++ -shared -fPIC ll_lunus_proxy.o -o liblunus_with_cub.so

clean:
	rm *.o *.so lunus_proxy
