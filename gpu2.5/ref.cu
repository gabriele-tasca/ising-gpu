#include <iostream> 
#include <cstdlib>
#include <cmath>
#define EPS 1e-6
#include <ctime>
using namespace std;


#define DEFAULT_SEED 777
#define NBLOCKS 16
#define DIMENSIONI 6
#define THREADS 32
#define NPUNTI 250000000
#include "MyMersenneTwister.h"


/* Using updated (v2) interfaces to cublas and cusparse */
#include <cuda_runtime_api.h>
#include <curand.h>
#include "curand_kernel.h"

// Utilities and system includes
#include <helper_functions.h>  // helper for shared functions common to CUDA SDK samples
#include <helper_cuda.h>       // helper for CUDA Error handling
#include <shrQATest.h>



// RNG init kernel
__global__ void initRNG(curandState * const rngStates,
                        const unsigned int seed)
{
    // Determine thread ID
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialise the RNG
    curand_init(seed, tid, 0, &rngStates[tid]);
}





////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void volumeSfera( curandState * const rngStates, dato *dev_c ){

  int  tid = threadIdx.x;
  int tidfull = blockIdx.x * blockDim.x + threadIdx.x;

  long int offset = DIMENSIONI*(blockDim.x*blockIdx.x+tid);
  long int maxpt = DIMENSIONI*(NPUNTI-1);
  float ran;
  dato r;
  __shared__ dato ir[THREADS];

  //  curandState_t state;
  // Initialise the RNG
  curandState localState = rngStates[tidfull];
//  curandState localState = rngStates[tid];


  long int temp = 0;
  while(offset<maxpt ){  

    r=0;  
    for(int i=0; i<DIMENSIONI;i++){
      ran= curand_uniform(&localState);
//    eventuale calcolo delle variabili cinematiche tramite kernel device

      r += pow(ran   ,2);

    };

//   eventuale calcolo del peso da associare alla configurazione in esame

//    valutazione di una osservabile (kernel device)  in questo esempio temp
    if (sqrt(r) <= 1)
      temp += 1;

    offset += blockDim.x*gridDim.x*DIMENSIONI;

  };
  ir[tid] = (dato)temp;

  __syncthreads();

  int i=blockDim.x/2;
  while(i!=0){

//     se ci fosse un peso calcolato per ciascuna configurazione del sistema
//     andrebbe introdotto nella sommatoria seguente
    if(tid<i)
      ir[tid]+=ir[tid+i];

    __syncthreads();
    i/=2;
  }
  if (tid==0)
    dev_c[blockIdx.x]=ir[0];


};



/////////////////////////////////////////////////////////////////
int main(void) {


  dato integrale;
  dato *partial_c;
  dato *dev_partial_c;
 
 
  partial_c = (dato *)malloc(NBLOCKS*sizeof(dato));
  cudaMalloc( (void **)&dev_partial_c, NBLOCKS*sizeof(dato) );

    // Determine how to divide the work between cores
    dim3 block;
    dim3 grid;
    //    block.x = threadBlockSize;
    //grid.x  = (numSims + threadBlockSize - 1) / threadBlockSize;
    block.x = THREADS;
    grid.x  = (NBLOCKS + THREADS - 1) / THREADS;


    // Allocate memory for RNG states
    curandState *d_rngStates = 0;
    cudaMalloc((void **)&d_rngStates, grid.x * block.x * sizeof(curandState));


    int seed = DEFAULT_SEED;
  // Initialise RNG
  initRNG<<<grid, block>>>(d_rngStates, seed);



  volumeSfera<<<NBLOCKS, THREADS>>>(d_rngStates, dev_partial_c);
  cudaMemcpy( partial_c, dev_partial_c, NBLOCKS*sizeof(dato), cudaMemcpyDeviceToHost);



  integrale=0.;
  //infine le somme parziali dei vari blocchi vengono combinate 
   for( int i=0; i<NBLOCKS; i++){
//     cout << i << "  " << partial_c[i]/NPUNTI <<"\n";
     integrale += partial_c[i];
   };
   integrale = integrale/NPUNTI*pow(2.,DIMENSIONI);
   cout << "integrale gpu=" << integrale <<"\n";

   cudaFree( dev_partial_c);

   free(partial_c);

	return 0;
};


////////////////////////////////////////////////////