module purge
module use /projects/icapt/mewall/modules
module load gcc/master
module load cuda/10.2

CC=gcc

if [ -e build ];then
  rm -rf build
fi

mkdir build
cd build

ln -s ../lunus_proxy.h .

export CFLAGS="-g -O3 -fopenmp -fPIC -DUSE_OPENMP -DUSE_OFFLOAD -DLUNUS_NUM_JBLOCKS=1 -DLUNUS_NUM_IBLOCKS=1 -DDEBUG -I. -L."

${CC} $CFLAGS -c ../llunus_proxy.c

if [[ $1 == "shared" ]];then
    ${CC} $CFLAGS -shared -o liblunus_proxy_shared.so llunus_proxy.o
    ${CC} $CFLAGS -o lunus_proxy ../lunus_proxy.c -llunus_proxy_shared
    LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH
elif [[ $1 == "static" ]];then
    ar rc liblunus_proxy_static.a llunus_proxy.o
    ${CC} $CFLAGS -o lunus_proxy ../lunus_proxy.c -llunus_proxy_static  
elif [[ $1 == "nolib" ]];then
    ${CC} $CFLAGS -o lunus_proxy ../lunus_proxy.c llunus_proxy.o
else # Assume static
    ar rc liblunus_proxy_static.a llunus_proxy.o
    ${CC} $CFLAGS -o lunus_proxy ../lunus_proxy.c -llunus_proxy_static  
fi
echo "Running application"
nvprof --print-gpu-trace ./lunus_proxy ../snc_newhead_00001.img out.img

cd -

#diff build/out.img ./out.img
#if [ $? == 0 ]; then
#  echo && echo "Test passed."
#else
#  echo && echo "Test failed."
#fi
