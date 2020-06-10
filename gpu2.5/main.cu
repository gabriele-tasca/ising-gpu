#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include "curand_kernel.h"
#include <assert.h>

// L should be  (multiple of (THR_NUMBER - 2) ) + 2
const int THR_NUMBER = 29;

#define SETBLOCKNUM 4
// #define L 122
const int L = (THR_NUMBER -2)* SETBLOCKNUM +2;

// #define MULTISPIN unsigned char
// #define MULTISIZE 8
#define MULTISPIN unsigned int
#define MULTISIZE 32


#define T_CYCLE_START 1.5
#define T_CYCLE_END 3.0
#define T_CYCLE_STEP 0.04

#define SINGLETEMP 2.4
int n_temps = ( T_CYCLE_END - T_CYCLE_START )/ (T_CYCLE_STEP);

#define J 1.

#define SEED 1000

const int AREA = L*L;
const int NTOT = (L-2)*(L-2);
// static const float EXP4_TRESHOLD = exp( -(4.*J) / T);
// static const float EXP8_TRESHOLD = exp( -(8.*J) / T);

#define STEPS_REPEAT 1
#define T_MAX_SIM 100
#define T_MEASURE_WAIT 50
#define T_MEASURE_INTERVAL 10

// print history true/false
#define HISTORY 1

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
// if the numbers overflow, then it would be necessary to divide by N before summing
// however it's faster the other way
void update_avg(struct avg_tr * tr_p, float newval) {
    tr_p->sum +=  newval;
    tr_p->sum_squares += (newval*newval);
}
float average( struct avg_tr tr) {
    return (tr.sum)/((float) tr.n) ;
}
float stdev( struct avg_tr tr) {
    return sqrt(  ( tr.sum_squares)/((float) tr.n)  -  pow(( (tr.sum)/((float) tr.n) ),2)  );
}
// float variance( struct avg_tr tr) {
//     return (  ( tr.sum_squares)/((float) tr.n)  -  pow(( (tr.sum)/((float) tr.n) ),2)  );
// }

//multispin averages, hard-coded to track a number MULTISPIN * STEPS_REPEAT of values
struct multiavg_tr {
    float sum[MULTISIZE * STEPS_REPEAT];
    float sum_squares[MULTISIZE * STEPS_REPEAT];
    int n; // number of terms in the avg
};
// localn is not multisize*steps_repeat, it's the number of terms that will contribute to each avg ...
struct multiavg_tr new_multiavg_tr(int localn) {
    struct multiavg_tr a;
    for(int k=0; k<MULTISIZE * STEPS_REPEAT; k++ ) {
        a.sum[k] = 0;
        a.sum_squares[k] = 0;
    }
    a.n = localn;
    return a;
}
// must be 0 =< k < MULTISIZE * STEPS_REPEAT
// void update_multiavg(struct multiavg_tr * tr_p, float newval, int k) {
//     tr_p->sum[k] +=  newval;
//     tr_p->sum_squares[k] += (newval*newval);
// }
__device__ void dev_update_multiavg(struct multiavg_tr * tr_p, float newval, int k) {
    tr_p->sum[k] +=  newval;
    tr_p->sum_squares[k] += (newval*newval);
}
float multiaverage( struct multiavg_tr tr, int k) {
    return (tr.sum[k])/((float) tr.n) ;
}
float multistdev( struct multiavg_tr tr, int k) {
    return sqrt(  ( tr.sum_squares[k])/((float) tr.n)  -  pow(( (tr.sum[k])/((float) tr.n) ),2)  );
}
// float multivariance( struct multiavg_tr tr, int k) {
//     return (  ( tr.sum_squares[k])/((float) tr.n)  -  pow(( (tr.sum[k])/((float) tr.n) ),2)  );
// }

// RNG init kernel
__global__ void initRNG(curandState * const rngStates, const int seed) {
    // Determine thread ID
    int blockId = blockIdx.x+ blockIdx.y * gridDim.x;
    int tid = blockId * (blockDim.x * blockDim.y)+ (threadIdx.y * blockDim.x)+ threadIdx.x;
    // Initialise the RNG
    curand_init(seed, tid, 0, &rngStates[tid]);
}


struct coords {
    int x;
    int y;
};
__device__ coords dev_get_thread_coords() {
    struct coords thread_coords;
 
    thread_coords.x =  blockIdx.x*( THR_NUMBER - 2 ) + ( threadIdx.x ) ;
    thread_coords.y =  blockIdx.y*( THR_NUMBER - 2 ) + ( threadIdx.y ) ;

    return thread_coords;
}

// float unitrand(){
//     return (float)rand() / (float)RAND_MAX;
// }
__device__ float dev_unitrand( curandState * const rngStates, unsigned int tid ){
    curandState localState = rngStates[tid];
    float val = curand_uniform(&localState);
    rngStates[tid] = localState;
    return val;
}

// index has to be less that MULTISIZE
__device__ void dev_set_spin_1 (MULTISPIN * multi, int index) {
    *multi |= 1 << index;
}
__device__ void dev_set_spin_0 (MULTISPIN * multi, int index) {
    *multi &= ~(1 << index);
}
__device__ MULTISPIN dev_read_spin(MULTISPIN multi, int index) {
     return ((multi >> index) & 1);
}
// each bit exp8 and exp8 describes the Metropolis RNG result for that bit,
// specifying if the random r is bigger or smaller than the relevant values e^(4J/kT) and e^(8J/kT) (passed from outside)
__device__ MULTISPIN generate_exp4_mask(float exp4, float exp8, curandState * const rngStates, int tid ) {
    MULTISPIN res;
    for(int k=0; k<MULTISIZE; k++) {
        float random_number = dev_unitrand(rngStates, tid); 
        if( exp4 > random_number && random_number > exp8) { // this is taken from the article and works. the version below might not but slightly simplifies some things
        // if( exp4 > random_number) {
            dev_set_spin_1(&res, k);
        } else {
            dev_set_spin_0(&res, k);
        }
    }
    return res;
}
__device__ MULTISPIN generate_exp8_mask(float exp8, curandState * const rngStates, int tid ) {
    MULTISPIN res;
    for(int k=0; k<MULTISIZE; k++) {
        float random_number = dev_unitrand(rngStates, tid); 
        if( random_number < exp8 ) {
            dev_set_spin_1(&res, k);
        } else {
            dev_set_spin_0(&res, k);
        }
    }
    return res;
}

MULTISPIN init_random_multispin() {
    return (MULTISPIN) rand(); // just spam random bits
}
void init_random_grid(MULTISPIN grid[L*L]) {
    for(int x = 0; x<L; x++) {
        for(int y = 0; y<L; y++) {
            grid[x+y*L] = init_random_multispin();
        }
    }
}


MULTISPIN init_t0_multispin() {
    return (MULTISPIN) 0; // should be all zeros for all sensible multispin types
}
void init_t0_grid(MULTISPIN grid[L*L]) {
    for(int x = 0; x<L; x++) {
        for(int y = 0; y<L; y++) {
            grid[x+y*L] = init_t0_multispin();
        }
    }
}

// void flip(MULTISPIN grid[L*L], int x, int y) {
//     grid[x+y*L] = ~grid[x+y*L];
// }

// can segfault 
__device__ static inline MULTISPIN dev_shared_grid_step(MULTISPIN shared_grid[THR_NUMBER*THR_NUMBER], int x, int y, int xstep, int ystep) {
    return shared_grid[(x+xstep) + (y+ystep)*THR_NUMBER];
}


// segfault if applied to an edge spin, must be called only on the inner L-1 grid
__device__ void dev_update_multispin_shared(MULTISPIN grid[THR_NUMBER*THR_NUMBER], int x, int y, float exp4, float exp8, curandState * const rngStates, int tid ) {

    MULTISPIN s0 = grid[x+y*THR_NUMBER];

    MULTISPIN exp4_mask = generate_exp4_mask(exp4, exp8, rngStates, tid ); // here
    MULTISPIN exp8_mask = generate_exp8_mask(exp8, rngStates, tid );

    // "energy variables" indicating whether s0 is equal or opposite to each of its 4 neighbours 
    MULTISPIN i1 = s0 ^ dev_shared_grid_step(grid, x, y, 1, 0);
    MULTISPIN i2 = s0 ^ dev_shared_grid_step(grid, x, y, -1, 0);
    MULTISPIN i3 = s0 ^ dev_shared_grid_step(grid, x, y, 0, 1);
    MULTISPIN i4 = s0 ^ dev_shared_grid_step(grid, x, y, 0, -1);
    
    // bit sums with carry over between the i variables
    MULTISPIN j1 = i1 & i2;
    MULTISPIN j2 = i1 ^ i2;
    MULTISPIN j3 = i3 & i4;
    MULTISPIN j4 = i3 ^ i4;

    // logic for deciding whether to flip s0 or not
    MULTISPIN flip_mask = ( ((j1 | j3) | (~(j1^j3) & (j2&j4))  )  |   ((j2 | j4) & exp4_mask  )   |   exp8_mask );

    grid[x+y*THR_NUMBER] = grid[x+y*THR_NUMBER] ^ flip_mask;


    // explanation:
    // spins | i1234 | deltaE | j1 j2  j3 j4 |
    //   1   |   1   |        |              |    
    //  101  |  1 1  |   -8   |  1 0    1 0  |                   
    //   1   |   1   |        |              |
    //                                                         
    //   0   |   0   |        |              |    
    //  101  |  1 1  |   -4   |  0 1    1 0  |         (j1 | j3)          
    //   1   |   1   |        |              |
    //                                                         
    //   0   |   0   |        |  0 0    1 0  |    
    //  001  |  0 1  |    0   |      or      |-------------------------                  
    //   1   |   1   |        |  0 1    0 1  |      ~(j1^j3) & (j2&j4))
    //------------------------------------------------------------------
    //                                                        
    //   0   |   0   |        |              |    
    //  000  |  0 0  |    +4  |              |       (j2 | j4) & exp4      
    //   1   |   1   |        |              |
    //------------------------------------------------------------------ 
    //                                                        
    //   0   |   0   |        |              |    
    //  000  |  0 0  |    +8  |  0 0    0 0  |           exp8       
    //   0   |   0   |        |              |

    // the first 2 cases are detected by (j1 | j3) and lead to the spin flip regardless of the RNG roll.
    // the deltaH = 0 case can result in two different forms for the j's depending on ho the spins are paired. 
    //   the first of these is correctly picked up by (j1 | j3), while the second needs its own expression ~(j1^j3) & (j2&j4))
    // in the 4th case, detected by (j2 | j4), the spin is flipped only if the RNG roll is lucky enough (exp4 = 1)
    // if we still haven't flipped, we get to the last case. here the spin is flipped only if the RNG roll gives the luckiest result (exp8 = 1).
    
}



// for now with nthreads = NTOT
__device__ void dev_update_grid(MULTISPIN grid[L*L], float exp4, float exp8, curandState * const rngStates ) {
    // the first argument here is the GLOBAL grid
    
    // thread coords relative to the GLOBAL grid
    struct coords glob_coords = dev_get_thread_coords();
    int glob_x = glob_coords.x;
    int glob_y = glob_coords.y;

    // Determine thread ID (for RNG)
    int blockId = blockIdx.x+ blockIdx.y * gridDim.x;
    int tid = blockId * (blockDim.x * blockDim.y)+ (threadIdx.y * blockDim.x)+ threadIdx.x;


    __shared__ MULTISPIN shared_grid[ THR_NUMBER*THR_NUMBER ];

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
                dev_update_multispin_shared( shared_grid, shared_x, shared_y, exp4, exp8, rngStates, tid);
            }
        }
        __syncthreads();

        if ( threadIdx.x != 0 && threadIdx.x != THR_NUMBER-1 && 
            threadIdx.y != 0 && threadIdx.y != THR_NUMBER-1 ) {
            // black
            if( (glob_x + glob_y%2)%2 == 1 ) {
                dev_update_multispin_shared( shared_grid, shared_x, shared_y, exp4, exp8, rngStates, tid);
            }
        }
        __syncthreads();

        if ( threadIdx.x > 0 && threadIdx.x != THR_NUMBER-1 && 
            threadIdx.y > 0 && threadIdx.y != THR_NUMBER-1 ) {
            grid[(glob_x )+ (glob_y )*L ]  = shared_grid[ threadIdx.x + threadIdx.y*THR_NUMBER ] ; 
        }
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
                dev_update_multispin_shared( shared_grid, shared_x, shared_y, exp4, exp8, rngStates, tid);
            }
        }
        __syncthreads();

        if ( threadIdx.x != 0 && threadIdx.x != THR_NUMBER-1 && 
            threadIdx.y != 0 && threadIdx.y != THR_NUMBER-1 ) {
            // black
            if( (glob_x + glob_y%2)%2 == 1 ) {
                dev_update_multispin_shared( shared_grid, shared_x, shared_y, exp4, exp8, rngStates, tid);
            }
        }
        __syncthreads();

        if ( threadIdx.x > 0 && threadIdx.x != THR_NUMBER-1 && 
            threadIdx.y > 0 && threadIdx.y != THR_NUMBER-1 ) {
            grid[(glob_x )+ (glob_y )*L ]  = shared_grid[ threadIdx.x + threadIdx.y*THR_NUMBER ] ; 
        }
        //////////
    }

}

// non GPU function
void multidump_first(MULTISPIN grid[L*L]) {
    // printf("first bit grid (out of %i):\n", MULTISIZE);
    for(int x = 0; x<L; x++) {
        for(int y = 0; y<L; y++) {
            
            if(( grid[x+y*L] & 1 ) == 0) printf(" ");
            else printf("█");

        }
        printf("\n");
    }
    printf("\n");
}

// non GPU function
void multidump_a_few(MULTISPIN grid[L*L]) {
    for(int k=0; k<5; k++) {
        printf("grid on bit %i (out of %i):\n", k+1, MULTISIZE);
        for(int x = 0; x<L; x++) {
            for(int y = 0; y<L; y++) {
                
                if(( grid[x+y*L] & (1 << k) ) == 0) printf(" ");
                else printf("█");

            }
            printf("\n");
        }
        printf("\n");
    }
}

// // GPU with nthreads = NTOT

// // as above, the number of independent measures is hardcoded to MULTISIZE * STEPS_REPEAT.
// // this function measures a single multispin, updating a chunk of the tracker. (for example the first half if rep_steps is 2)
__device__ void dev_update_magnetization_tracker(MULTISPIN dev_grid[L*L], struct multiavg_tr * dev_tr_p, int * dev_partial_res, int rep_count ) {
    int y = (  threadIdx.x / (L-2) ) +1;
    int x = (  threadIdx.x % (L-2) ) +1;
    for( int multik=0; multik < MULTISIZE; multik++) {
        float spin = (float) dev_read_spin(dev_grid[x+y*L] , multik); 
        atomicAdd(dev_partial_res, (spin*2.)-1.  );
        __syncthreads();
        
        if (threadIdx.x == 0) {
            *dev_partial_res = *dev_partial_res / (float) NTOT;
            dev_update_multiavg( dev_tr_p, *dev_partial_res, multik + MULTISIZE*rep_count );
            *dev_partial_res = 0;
        }
        __syncthreads();
    }
}

__global__ void dev_measure_cycle_kernel(MULTISPIN * dev_grid, curandState * const rngStates, struct multiavg_tr * dev_single_run_avgs, int * dev_partial_res, float exp4, float exp8  ) {

    // INNER SIM LOOPS
    int ksim=0;
    for( ; ksim<T_MEASURE_WAIT; ksim++) {
        dev_update_grid(dev_grid, exp4, exp8, rngStates );
    }
    // __syncthreads();
    
    for( ; ksim<T_MAX_SIM; ksim++) {
        dev_update_grid(dev_grid, exp4, exp8, rngStates );
        
        if( ksim % T_MEASURE_INTERVAL == 0) {
            // dev_update_magnetization_tracker(dev_grid, dev_single_run_avgs, dev_partial_res, krep);                
        }
        // __syncthreads();
    }
    // END INNER SIM LOOPS        

    
    // INNER SIM LOOPS

    // int ksim=0;
    // for( ; ksim<pl.t_measure_wait; ksim++) {
    //     dev_update_grid_shared(dev_grid, rngStates, temperature);
    // }
    // // end thermalization

    // for( ; ksim<pl.t_max_sim; ksim++) {
    //     dev_update_grid_shared(dev_grid, rngStates, temperature);

    //     ////////////measures
    //     if( ksim % pl.t_measure_interval == 0) {
    //         dev_update_magnetization_tracker(dev_grid, dev_single_run_avg, dev_partial_res );
    //     }

    // }
    // // END INNER SIM LOOPS
}

void parall_measure_cycle(MULTISPIN startgrid[L*L], MULTISPIN * dev_grid, float exp4, float exp8, curandState * const rngStates, FILE *resf) {


    float n_measures_per_sim = (float) ((T_MAX_SIM - T_MEASURE_WAIT)/T_MEASURE_INTERVAL);

    //OUTER REP LOOP  
    struct multiavg_tr single_run_avgs = new_multiavg_tr(n_measures_per_sim);
    struct multiavg_tr * dev_single_run_avgs;
    cudaMalloc(&dev_single_run_avgs, sizeof(struct multiavg_tr));
    cudaMemcpy(dev_single_run_avgs, &single_run_avgs, sizeof(struct multiavg_tr), cudaMemcpyHostToDevice);

    // extra space needed by update_magnetization
    int * dev_partial_res;
    cudaMalloc(&dev_partial_res, sizeof(int));

    for( int krep=0; krep< STEPS_REPEAT; krep++) {
        if (HISTORY) printf("# simulation %i\n", krep+1);
        if (HISTORY) printf("#    waiting thermalization for the first %i sim steps.\n", T_MEASURE_WAIT);

        cudaMemcpy(dev_grid, startgrid, L*L*sizeof(MULTISPIN), cudaMemcpyHostToDevice);
        
        dev_measure_cycle_kernel<<<BLOCKS, THREADS>>>(dev_grid, rngStates, dev_single_run_avgs, dev_partial_res, exp4, exp8 );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("kernel: ERROR: %s\n", cudaGetErrorString(err));
        } else printf("kernel: no ERROR: %s\n", cudaGetErrorString(err));

        if (HISTORY) printf("# end simulation %i\n", krep+1);
    }
    // END OUTER REPETITION LOOP



    cudaMemcpy(&single_run_avgs, dev_single_run_avgs, sizeof(struct multiavg_tr), cudaMemcpyDeviceToHost);
    
    ///////////////
    struct avg_tr avg_of_runs = new_avg_tr( MULTISIZE * STEPS_REPEAT );
    for(int k=0; k < MULTISIZE * STEPS_REPEAT; k++) {
        float lres = multiaverage(single_run_avgs, k);
        float lstdev = multistdev(single_run_avgs, k);

        fprintf(resf, "# average of simulation %i\n: %f +- %f\n", k+1, lres, lstdev);
        update_avg(&avg_of_runs, lres);
    }
    fprintf(resf, "# overall average \n: %f +- %f\n", average(avg_of_runs), stdev(avg_of_runs));
    ////////////////

    // grid for displaying end-state (of last rep only)
    MULTISPIN endgrid[L*L];
    cudaMemcpy(endgrid, dev_grid, L*L*sizeof(MULTISPIN), cudaMemcpyDeviceToHost);

    if (HISTORY) multidump_first(endgrid);

    cudaFree(dev_partial_res);
    cudaFree(dev_single_run_avgs);

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
    fprintf(resf, "# coupling: %f\n# repetitions: %i\n", J, STEPS_REPEAT);
    fprintf(resf, "# simulation_t_max: %i\n# thermalization_time: %i\n# time_between_measurements: %i\n# base_random_seed: %i\n",  T_MAX_SIM,T_MEASURE_WAIT, T_MEASURE_INTERVAL, SEED);
    fprintf(resf, "# extra:\n# area: %i\n# active_spins_excluding_boundaries:%i\n", AREA, NTOT);
    fprintf(resf, "\n");
    fprintf(resf, "# columns: temperature - average magnetization - uncertainty \n");
    
    // still used for init_random_grid
    srand(SEED);

    // curand init
    // Allocate memory for RNG states
    curandState *d_rngStates = 0;

    cudaMalloc((void **)&d_rngStates, THR_NUMBER*THR_NUMBER*BLOCK_NUMBER*BLOCK_NUMBER*sizeof(curandState));
    // Initialise RNG
    initRNG<<<BLOCKS, THREADS>>>(d_rngStates, SEED);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("initRNG: ERROR: %s\n", cudaGetErrorString(err));
    } else printf("initRNG: no ERROR: %s\n", cudaGetErrorString(err));
     

    // device grid
    MULTISPIN * dev_grid;
    cudaMalloc(&dev_grid, L*L*sizeof(MULTISPIN));

    // original grid on the cpu
    MULTISPIN startgrid[L*L];
    init_t0_grid(startgrid);
    // multidump_a_few(startgrid);

    // // temp cycle:
    // for( float kt=T_CYCLE_START; kt<T_CYCLE_END; kt+=T_CYCLE_STEP ) {
    //     const float EXP4 = exp( -(4.*J) / kt);
    //     const float EXP8 = exp( -(8.*J) / kt);

    //     parall_measure_cycle(startgrid, dev_grid, EXP4, EXP8, d_rngStates, resf);
    // }

    // only 1:
    const float EXP4 = exp( -(4.*J) / SINGLETEMP);
    const float EXP8 = exp( -(8.*J) / SINGLETEMP);
    parall_measure_cycle(startgrid, dev_grid, EXP4, EXP8, d_rngStates, resf);


    cudaFree(d_rngStates);
    cudaFree(dev_grid);


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float total_time = 0;
    cudaEventElapsedTime(&total_time, start, stop);

    FILE *timef = fopen("time.txt", "w");
    long int total_flips = ((long int)(n_temps))* ((long int)((STEPS_REPEAT))) * ((long int)(T_MAX_SIM)) * ((long int)(NTOT));
    
    fprintf(timef, "# gpu1\n");
    fprintf(timef, "# total execution time (milliseconds):\n");
    fprintf(timef, "%f\n", total_time);
    fprintf(timef, "# total spin flips performed:\n");
    fprintf(timef, "%li\n", total_flips);
    fprintf(timef, "# average spin flips per millisecond:\n");
    fprintf(timef, "%Lf\n", ((long double) total_flips  )/( (long double) total_time ) );

    fclose(timef);

    fclose(resf);

    
    return 0;
}

