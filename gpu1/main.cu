#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include "curand_kernel.h"
#include <assert.h>

// L should be  (multiple of (THR_NUMBER - 2) ) + 2
#define L 83

const int AREA = L*L;
const int NTOT = (L-2)*(L-2);

// #define T 6.
// #define T 0.1
// #define T 2.26918531421
#define T_CYCLE_START 1.6
#define T_CYCLE_END 2.9
#define T_CYCLE_STEP 0.04

#define SINGLETEMP 2.4

int n_temps = ( T_CYCLE_END - T_CYCLE_START )/ (T_CYCLE_STEP);

#define J 1.
#define SEED 1000

struct measure_plan {
    int steps_repeat;
    int t_max_sim;
    int t_measure_wait;
    int t_measure_interval; } 
static PLAN = {
    .steps_repeat = 20,
    .t_max_sim = 200,
    .t_measure_wait = 50,
    .t_measure_interval = 20  };


// print history true/false
#define HISTORY 1

const int THR_NUMBER = 29;
const int BLOCK_NUMBER  = ( L-2)/( THR_NUMBER - 2 );
const dim3 BLOCKS( BLOCK_NUMBER, BLOCK_NUMBER );
const dim3 THREADS( THR_NUMBER, THR_NUMBER );

// average tracker struct 
struct avg_tr {
    float sum;
    float sum_squares;
    int n;
};
struct avg_tr new_avg_tr(int locn) {
    struct avg_tr a = { .sum = 0, .sum_squares = 0, .n = locn};
    return a;
}
void update_avg(struct avg_tr * tr_p, float newval) {
    tr_p->sum +=  newval;
    tr_p->sum_squares += (newval*newval);
}
__device__ static inline void dev_update_avg(struct avg_tr * tr_p, float newval) {
    tr_p->sum +=  newval;
    tr_p->sum_squares += (newval*newval);
}
float average( struct avg_tr tr) {
    return (tr.sum)/((float) tr.n) ;
}
float stdev( struct avg_tr tr) {
    return sqrt(  ( tr.sum_squares)/((float) tr.n)  -  pow(( (tr.sum)/((float) tr.n) ),2)  );
}
float variance( struct avg_tr tr) {
    return (  ( tr.sum_squares)/((float) tr.n)  -  pow(( (tr.sum)/((float) tr.n) ),2)  );
}

// RNG init kernel
__global__ void initRNG(curandState * const rngStates, const unsigned int seed) {
    // Determine thread ID
    int blockId = blockIdx.x+ blockIdx.y * gridDim.x;
    int tid = blockId * (blockDim.x * blockDim.y)+ (threadIdx.y * blockDim.x)+ threadIdx.x;
    // Initialise the RNG
    curand_init(seed, tid, 0, &rngStates[tid]);
}

// static inline float unitrand(){
//     return (float)rand() / (float)RAND_MAX;
// }
__device__ static inline float dev_unitrand( curandState * const rngStates, unsigned int tid ){
    curandState localState = rngStates[tid];
    float val = curand_uniform(&localState);
    rngStates[tid] = localState;
    return val;
}

void init_random(char grid[L*L]) {
    for(int x = 0; x<L; x++) {
        for(int y = 0; y<L; y++) {
            grid[x+y*L] = rand() & 1;
        }
    }
}
void init_t0(char grid[L*L]) {
    for(int x = 0; x<L; x++) { 
        for(int y = 0; y<L; y++) {
            grid[x+y*L] = 0;
        }
    }
}

void dump(char grid[L*L]) {
    for(int x = 0; x<L; x++) {
        for(int y = 0; y<L; y++) {
            // if(grid[x+y*L] == 0) printf("•");
            // else printf("◘");
            if(grid[x+y*L] == 0) printf(" ");
            else printf("█");
            // printf("%i", grid[x+y*L]);
        }
        printf("\n");
    }
    printf("\n");
}
__device__ void dev_dump(char grid[L*L]) {
    for(int x = 0; x<L; x++) {
        for(int y = 0; y<L; y++) {
            // if(grid[x+y*L] == 0) printf("•");
            // else printf("◘");
            if(grid[x+y*L] == 0) printf(" ");
            else printf("█");
            // printf("%i", grid[x+y*L]);
        }
        printf("\n");
    }
    printf("\n");
}

struct coords {
    int x;
    int y;
};
__device__ static inline coords dev_get_thread_coords() {
    struct coords thread_coords;
    // the 4 lines below are outdated haha
    // assign loc_x and loc_y so that only the inner square is covered
    // also remember that now threads are launched in blocks of 32x32,
    // but only the inner 30x30 are mapped to the grid...
    // thread_coords on each block's edge threads mean nothing and should not be read
 
    thread_coords.x =  blockIdx.x*( THR_NUMBER - 2 ) + ( threadIdx.x ) ;
    thread_coords.y =  blockIdx.y*( THR_NUMBER - 2 ) + ( threadIdx.y ) ;

    return thread_coords;
}

// can segfault 
__device__ static inline char dev_shared_grid_step(char shared_grid[THR_NUMBER*THR_NUMBER], int x, int y, int xstep, int ystep) {
    return shared_grid[(x+xstep) + (y+ystep)*THR_NUMBER];
}

// segfault if applied to an edge spin, call only on the inner THR_NUMBER-1 grid
__device__ void dev_update_spin_shared(char dev_shared_grid[ THR_NUMBER*THR_NUMBER ], int x, int y , curandState * const rngStates, unsigned int tid, double temperature ) {
    char s0 = dev_shared_grid[x+y*THR_NUMBER];
    char j1 = s0 ^ dev_shared_grid_step(dev_shared_grid, x, y, 1, 0);
    char j2 = s0 ^ dev_shared_grid_step(dev_shared_grid, x, y, -1, 0);
    char j3 = s0 ^ dev_shared_grid_step(dev_shared_grid, x, y, 0, 1);
    char j4 = s0 ^ dev_shared_grid_step(dev_shared_grid, x, y, 0, -1);
    float dh = (float) ( -((j1 + j2 + j3 + j4) *2 -4)*2*J );

    float p = exp(  -dh / temperature);
    float ur = dev_unitrand(rngStates, tid);

    if(ur < p ) {
        dev_shared_grid[x+y*THR_NUMBER] = !dev_shared_grid[x+y*THR_NUMBER];
    } 
}


__device__ void dev_update_grid_shared(char grid[L*L], curandState * const rngStates, double temperature ) {
    // the first argument here is the GLOBAL grid

    // thread coords relative to the GLOBAL grid
    struct coords glob_coords = dev_get_thread_coords();
    int glob_x = glob_coords.x;
    int glob_y = glob_coords.y;

    // Determine thread ID (for RNG)
    int blockId = blockIdx.x+ blockIdx.y * gridDim.x;
    int tid = blockId * (blockDim.x * blockDim.y)+ (threadIdx.y * blockDim.x)+ threadIdx.x;

    __shared__ char shared_grid[ THR_NUMBER*THR_NUMBER ];

    shared_grid[ threadIdx.x + threadIdx.y*THR_NUMBER ] = grid[(glob_x )+ (glob_y )*L ]; // check formulas
    __syncthreads();

    // thread coords relative to the shared grid
    int shared_x = threadIdx.x;
    int shared_y = threadIdx.y;

    // macro-checkboards
    // macro-white
    if( (blockIdx.x + blockIdx.y%2)%2 == 0 ) {
        /////////////
        // checkboards
        // update only in the inner 30x30 block of threads, because the edge threads aren't mapped to any grid spins
        if ( threadIdx.x != 0 && threadIdx.x != THR_NUMBER-1 && 
            threadIdx.y != 0 && threadIdx.y != THR_NUMBER-1 ) {
            // white
            if( (glob_x + glob_y%2)%2 == 0 ) {
                dev_update_spin_shared( shared_grid, shared_x, shared_y, rngStates, tid, temperature );
            }
        }
        __syncthreads();

        if ( threadIdx.x != 0 && threadIdx.x != THR_NUMBER-1 && 
            threadIdx.y != 0 && threadIdx.y != THR_NUMBER-1 ) {
            // black
            if( (glob_x + glob_y%2)%2 == 1 ) {
                dev_update_spin_shared( shared_grid, shared_x, shared_y, rngStates, tid, temperature );
            }
        }
        __syncthreads();

        grid[(glob_x )+ (glob_y )*L ]  = shared_grid[ threadIdx.x + threadIdx.y*THR_NUMBER ] ; // check formulas
        //////////
    }
    __syncthreads();

    // macro-black
    if( (blockIdx.x + blockIdx.y%2)%2 == 1 ) {
        //////////

        // checkboards
        // update only in the inner 30x30 block of threads, because the edge threads aren't mapped to any grid spins
        if ( threadIdx.x != 0 && threadIdx.x != THR_NUMBER-1 && 
                threadIdx.y != 0 && threadIdx.y != THR_NUMBER-1 ) {
            // white
            if( (glob_x + glob_y%2)%2 == 0 ) {
                dev_update_spin_shared( shared_grid, shared_x, shared_y, rngStates, tid, temperature );
            }
        }
        __syncthreads();

        if ( threadIdx.x != 0 && threadIdx.x != THR_NUMBER-1 && 
            threadIdx.y != 0 && threadIdx.y != THR_NUMBER-1 ) {
            // black
            if( (glob_x + glob_y%2)%2 == 1 ) {
                dev_update_spin_shared( shared_grid, shared_x, shared_y, rngStates, tid, temperature );
            }
        }
        __syncthreads();

        grid[(glob_x )+ (glob_y )*L ]  = shared_grid[ threadIdx.x + threadIdx.y*THR_NUMBER ] ; // check formulas

        //////////
    }

}


__device__ void dev_update_magnetization_tracker(char dev_grid[L*L], float * dev_single_run_avg, int * dev_partial_res ) {
    struct coords glob_coords = dev_get_thread_coords();
    int glob_x = glob_coords.x;
    int glob_y = glob_coords.y;

    if ( threadIdx.x != 0 && threadIdx.x != THR_NUMBER-1 && 
        threadIdx.y != 0 && threadIdx.y != THR_NUMBER-1 ) {
        int spin = (int) dev_grid[glob_x+glob_y*L]; 
        atomicAdd(dev_partial_res, spin );
    }
    __syncthreads();
    
    if ( blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        float val = ( ((float)  (*dev_partial_res) *2 ) - NTOT    ) / (float) NTOT;
        /*AAAAA*/ *dev_single_run_avg += val;
        *dev_partial_res = 0;
    }
        
}

__global__ /*AAAAA*/ void dev_measure_cycle_kernel(struct measure_plan pl, char * dev_grid, curandState * const rngStates, float * dev_single_run_avg, int * dev_partial_res , double temperature ) {
    // INNER SIM LOOPS

    int ksim=0;
    for( ; ksim<pl.t_measure_wait; ksim++) {
        dev_update_grid_shared(dev_grid, rngStates, temperature);
    }
    // end thermalization

    for( ; ksim<pl.t_max_sim; ksim++) {
        dev_update_grid_shared(dev_grid, rngStates, temperature);

        ////////////measures
        if( ksim % pl.t_measure_interval == 0) {
            dev_update_magnetization_tracker(dev_grid, dev_single_run_avg, dev_partial_res );
        }

    }
    // END INNER SIM LOOPS
}

void parall_measure_cycle(char startgrid[L*L], struct measure_plan pl, char * dev_grid, curandState * const rngStates, FILE *resf, double temperature ) {



    //OUTER REP LOOP
    ////////////measures
    float n_measures_per_sim = (float) ((pl.t_max_sim - pl.t_measure_wait)/pl.t_measure_interval);
    
    struct avg_tr outer_avg_tr = new_avg_tr(pl.steps_repeat);
    

    // extra space needed by dev_update_magnetization_tracker
    int * dev_partial_res;
    cudaMalloc(&dev_partial_res, sizeof(int));


    for( int krep=0; krep< pl.steps_repeat; krep++) {
        
        /*AAAAA*/ float single_run_avg = 0.;
        /*AAAAA*/ float * dev_single_run_avg;
        /*AAAAA*/ cudaMalloc(&dev_single_run_avg, sizeof(float));
        /*AAAAA*/ cudaMemcpy(dev_single_run_avg, &single_run_avg, sizeof(float), cudaMemcpyHostToDevice);

        // printf("seeding with %i\n", SEED+krep);
        // initialize starting grid on the device for this sim
        cudaMemcpy(dev_grid, startgrid, L*L*sizeof(char), cudaMemcpyHostToDevice);
  
        /*AAAAA*/ dev_measure_cycle_kernel<<<BLOCKS, THREADS>>>(pl, dev_grid, rngStates, dev_single_run_avg, dev_partial_res, temperature );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("ERROR: %s\n", cudaGetErrorString(err));
        }

        // bring back results to CPU
        /*AAAAA*/ cudaMemcpy(&single_run_avg, dev_single_run_avg, sizeof(float), cudaMemcpyDeviceToHost);
        /*AAAAA*/ float lres = single_run_avg/(n_measures_per_sim);
        // /*AAAAA*/ float lstdev = stdev(single_run_avg);
        if (HISTORY) printf(" temperature: %f\n", temperature);
        if (HISTORY) printf("# average of simulation %i:\n %f\n", krep+1, lres);

        update_avg(&outer_avg_tr, lres);
        
        char endgrid[L*L];
        cudaMemcpy(endgrid, dev_grid, L*L*sizeof(char), cudaMemcpyDeviceToHost);
        if (HISTORY) dump(endgrid);
        
        /*AAAAA*/ cudaFree(dev_single_run_avg);
    
    }

    // END OUTER REP LOOP
    
    ////////////measures
    fprintf(resf, "%f ", temperature);
    fprintf(resf, "%f ", average(outer_avg_tr));
    fprintf(resf, "%f\n", stdev(outer_avg_tr));
    
    cudaFree(dev_partial_res);

}



int main() {
    // L should be  (multiple of THR_NUMBER -2) + 2
    assert( ((L-2)% (THR_NUMBER-2)  )== 0 );

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);


    FILE *resf = fopen("results.txt", "w");
    fprintf(resf, "# gpu1\n");
    fprintf(resf, "# parameters:\n# linear_size: %i\n", L);
    fprintf(resf, "# coupling: %f\n# repetitions: %i\n", J, PLAN.steps_repeat);
    fprintf(resf, "# simulation_t_max: %i\n# thermalization_time: %i\n# time_between_measurements: %i\n# base_random_seed: %i\n",  PLAN.t_max_sim, PLAN.t_measure_wait, PLAN.t_measure_interval, SEED);
    fprintf(resf, "# extra:\n# area: %i\n# active_spins_excluding_boundaries:%i\n", AREA, NTOT);
    fprintf(resf, "\n");
    fprintf(resf, "# columns: temperature - average magnetization - uncertainty \n");

    srand(SEED);

    // curand init
    // Allocate memory for RNG states
    curandState *d_rngStates = 0;

    cudaMalloc((void **)&d_rngStates, THR_NUMBER*THR_NUMBER*BLOCK_NUMBER*BLOCK_NUMBER*sizeof(curandState));
    // Initialise RNG
    initRNG<<<BLOCKS, THREADS>>>(d_rngStates, SEED);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: %s\n", cudaGetErrorString(err));
    }

    // device grid
    char * dev_grid;
    cudaMalloc(&dev_grid, L*L*sizeof(char));

    char startgrid[L*L];
    init_t0(startgrid); 
    // if (HISTORY) printf("starting grid:\n");
    // if (HISTORY) dump(startgrid);

    
    // // temp cycle:
    for( double kt=T_CYCLE_START; kt<T_CYCLE_END; kt+=T_CYCLE_STEP ) {
        parall_measure_cycle(startgrid, PLAN, dev_grid, d_rngStates, resf, kt);
    }

    // only 1:
    parall_measure_cycle(startgrid, PLAN, dev_grid, d_rngStates, resf, SINGLETEMP);
        

    cudaFree(d_rngStates);
    cudaFree(dev_grid);


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float total_time = 0;
    cudaEventElapsedTime(&total_time, start, stop);

    FILE *timef = fopen("time.txt", "w");
    int total_flips = n_temps * PLAN.steps_repeat * PLAN.t_max_sim * NTOT;
    fprintf(timef, "# total execution time (milliseconds):\n");
    fprintf(timef, "%f\n", total_time);
    fprintf(timef, "# total spin flips performed:\n");
    fprintf(timef, "%f\n", total_flips);
    fprintf(timef, "# average spin flips per millisecond:\n");
    fprintf(timef, "%f\n", ((float) total_flips  )/( (float) total_time ) );

    fclose(timef);

    fclose(resf);

}

