#include <math.h>
#include <stdio.h>

#include <cuda_runtime_api.h>
#include <curand.h>
#include "curand_kernel.h"

# define SEED 1000
# define NTOT 10

// RNG init kernel
__global__ void initRNG(curandState * const rngStates, const unsigned int seed) {
    // Determine thread ID
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Initialise the RNG
    curand_init(seed, tid, 0, &rngStates[tid]);
    if (threadIdx.x == 0) for(int i=0; i<1000; i++)      printf("from init rng\n");

}

__device__ static inline float dev_unitrand( curandState * const rngStates, unsigned int tid ){
    curandState localState = rngStates[tid];
    return curand_uniform(&localState);
}

__global__ void spam_rands(curandState * const rngStates, unsigned int tid ) {
    printf("le aoeufbakhfb\n");
    for(int i=0; i<10; i++) {


        float ur = dev_unitrand(rngStates, tid); 
        if (threadIdx.x == 0)  printf("unitrand: %f \n", ur);
        if (threadIdx.x == 1)  printf("unitrand: %f \n", ur);
        if (threadIdx.x == 2)  printf("unitrand: %f \n", ur);
        
    } 
}


int main() {
    // curand init
    // Allocate memory for RNG states
    curandState *d_rngStates = 0;
    // cudaMalloc((void **)&d_rngStates, grid.x * block.x * sizeof(curandState));
    cudaMalloc((void **)&d_rngStates, NTOT*sizeof(curandState));
    // Initialise RNG
    initRNG<<<1, NTOT>>>(d_rngStates, SEED);
    printf("le edge\n");
    spam_rands<<<1, NTOT>>>(d_rngStates, SEED);



   
    cudaFree(&d_rngStates);
}

