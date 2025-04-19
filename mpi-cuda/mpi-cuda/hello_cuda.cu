
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

__global__ void cuda_hello(){
    // thread id of current block (on x axis)
    int tid = threadIdx.x;

    int bid = blockIdx.x;
    int bx = blockDim.x;

    int gid = bx*bid + tid; 
    printf("Ciao belli from core %d-%d global index %d!\n", bid, tid, gid);
}

extern "C" void launch_cuda() {
    // Launch GPU kernel
    cuda_hello<<<2,4>>>();

    // cuda synch barrier
    cudaDeviceSynchronize();

}
