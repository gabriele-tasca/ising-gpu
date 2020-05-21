#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include <cuda_runtime_api.h>
#include <curand.h>
#include "curand_kernel.h"

#define L 10


#define MULTISPIN unsigned char
#define MULTISIZE 8


// #define T 6.
// #define T 0.01
#define T 2.26918531421

#define J 1.

// #define MULTISPIN unsigned long int
// #define MULTISIZE 64


#define SEED 1002

const int AREA = L*L;
const int NTOT = (L-2)*(L-2);
// static const float EXP4_TRESHOLD = exp( -(4.*J) / T);
// static const float EXP8_TRESHOLD = exp( -(8.*J) / T);

#define STEPS_REPEAT 2
#define T_MAX_SIM 100
#define T_MEASURE_WAIT 50
#define T_MEASURE_INTERVAL 5

struct params {
    float J,

    int seed,
    int steps_repeat;
    int t_max_sim;
    int t_measure_wait;
    int t_measure_interval; }
// const PLAN = {
//     .steps_repeat = STEPS_REPEAT,
//     .t_max_sim = T_MAX_SIM,
//     .t_measure_wait = T_MEASURE_WAIT,
//     .t_measure_interval = T_MEASURE_INTERVAL  };


// average tracker struct
struct avg_tr {
    float sum;
    float sum_squares;
    int n;
};
static inline struct avg_tr new_avg_tr(int locn) {
    struct avg_tr a = { .sum = 0, .sum_squares = 0, .n = locn};
    return a;
}
// if the numbers overflow, then it would be necessary to divide by N before summing
// however it's faster the other way
static inline void update_avg(struct avg_tr * tr_p, float newval) {
    tr_p->sum +=  newval;
    tr_p->sum_squares += (newval*newval);
}
static inline float average( struct avg_tr tr) {
    return (tr.sum)/((float) tr.n) ;
}
static inline float stdev( struct avg_tr tr) {
    return sqrt(  ( tr.sum_squares)/((float) tr.n)  -  pow(( (tr.sum)/((float) tr.n) ),2)  );
}
// static inline float variance( struct avg_tr tr) {
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
// static inline void update_multiavg(struct multiavg_tr * tr_p, float newval, int k) {
//     tr_p->sum[k] +=  newval;
//     tr_p->sum_squares[k] += (newval*newval);
// }
__device__ static inline void dev_update_multiavg(struct multiavg_tr * tr_p, float newval, int k) {
    tr_p->sum[k] +=  newval;
    tr_p->sum_squares[k] += (newval*newval);
}
static inline float multiaverage( struct multiavg_tr tr, int k) {
    return (tr.sum[k])/((float) tr.n) ;
}
static inline float multistdev( struct multiavg_tr tr, int k) {
    return sqrt(  ( tr.sum_squares[k])/((float) tr.n)  -  pow(( (tr.sum[k])/((float) tr.n) ),2)  );
}
// static inline float multivariance( struct multiavg_tr tr, int k) {
//     return (  ( tr.sum_squares[k])/((float) tr.n)  -  pow(( (tr.sum[k])/((float) tr.n) ),2)  );
// }

// RNG init kernel
__global__ void initRNG(curandState * const rngStates, const unsigned int seed) {
    // Determine thread ID
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Initialise the RNG
    curand_init(seed, tid, 0, &rngStates[tid]);
}


// static inline float unitrand(){
//     return (float)rand() / (float)RAND_MAX;
// }
__device__ static inline float dev_unitrand( curandState * const rngStates, int tid ){
    curandState localState = rngStates[tid];
    return curand_uniform(&localState);
}

// index has to be less that MULTISIZE
__device__ static inline void dev_set_spin_1 (MULTISPIN * multi, int index) {
    *multi |= 1 << index;
}
__device__ static inline void dev_set_spin_0 (MULTISPIN * multi, int index) {
    *multi &= ~(1 << index);
}
__device__ static inline MULTISPIN dev_read_spin(MULTISPIN multi, int index) {
     return ((multi >> index) & 1);
}
// each bit exp8 and exp8 describes the Metropolis RNG result for that bit,
// specifying if the random r is bigger or smaller than the relevant values e^(4J/kT) and e^(8J/kT) (passed from outside)
__device__ static inline MULTISPIN generate_exp4_mask(float exp4, float exp8, float random_number) {
    MULTISPIN res;
    for(int k=0; k<MULTISIZE; k++) {
        if( exp4 > random_number && random_number > exp8) { // this is taken from the article and works. the version below might not but slightly simplifies some things
        // if( exp4 > random_number) {
            dev_set_spin_1(&res, k);
        } else {
            dev_set_spin_0(&res, k);
        }
    }
    return res;
}
__device__ static inline MULTISPIN generate_exp8_mask(float exp8, float random_number) {
    MULTISPIN res;
    for(int k=0; k<MULTISIZE; k++) {
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

// static inline void flip(MULTISPIN grid[L*L], int x, int y) {
//     grid[x+y*L] = ~grid[x+y*L];
// }

// can segfault 
__device__ static inline MULTISPIN grid_step(MULTISPIN grid[L*L], int x, int y, int xstep, int ystep) {
    return grid[(x+xstep)  + (y+ystep)*L];
}


// segfault if applied to an edge spin, must be called only on the inner L-1 grid
__device__ void dev_update_multispin(MULTISPIN grid[L*L], int x, int y, float exp4, float exp8, curandState * const rngStates, int tid ) {

    MULTISPIN s0 = grid[x+y*L];

    float ur = dev_unitrand(rngStates, tid); 
    MULTISPIN exp4_mask = generate_exp4_mask(exp4, exp8, ur); // here
    MULTISPIN exp8_mask = generate_exp8_mask(exp8, ur);

    // "energy variables" indicating whether s0 is equal or opposite to each of its 4 neighbours 
    MULTISPIN i1 = s0 ^ grid_step(grid, x, y, 1, 0);
    MULTISPIN i2 = s0 ^ grid_step(grid, x, y, -1, 0);
    MULTISPIN i3 = s0 ^ grid_step(grid, x, y, 0, 1);
    MULTISPIN i4 = s0 ^ grid_step(grid, x, y, 0, -1);
    
    // bit sums with carry over between the i variables
    MULTISPIN j1 = i1 & i2;
    MULTISPIN j2 = i1 ^ i2;
    MULTISPIN j3 = i3 & i4;
    MULTISPIN j4 = i3 ^ i4;

    // logic for deciding whether to flip s0 or not
    MULTISPIN flip_mask = ( ((j1 | j3) | (~(j1^j3) & (j2&j4))  )  |   ((j2 | j4) & exp4_mask  )   |   exp8_mask );

    grid[x+y*L] = grid[x+y*L] ^ flip_mask;

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
__global__ void dev_update_grid(MULTISPIN grid[L*L], float exp4, float exp8, curandState * const rngStates ) {
    // assign loc_x and loc_y so that only the inner square is covered
    int loc_y = (  threadIdx.x / (L-2) ) +1;
    int loc_x = (  threadIdx.x % (L-2) ) +1;

    int tid = threadIdx.x;    // change

    // white
    if( (loc_x + loc_y%2)%2 == 0 ) {
        dev_update_multispin( grid, loc_x, loc_y, exp4, exp8, rngStates, tid );
    }
    __syncthreads();
    // black
    if( (loc_x + loc_y%2)%2 == 1 ) {
        dev_update_multispin( grid, loc_x, loc_y, exp4, exp8, rngStates, tid );
    }
    __syncthreads();
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
__global__ void dev_update_magnetization_tracker(MULTISPIN dev_grid[L*L], struct multiavg_tr * dev_tr_p, float * dev_partial_res, int rep_count ) {
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


void parall_measure_cycle(MULTISPIN startgrid[L*L], float exp4, float exp8, curandState * const rngStates) {
    FILE *resf = fopen("results.txt", "w");
    fprintf(resf, "# hard-coded parameters:\n# linear_size: %i\n# spin_coding_size: %i\n", L, MULTISIZE);
    fprintf(resf, "# parameters:\n# temperature: %f\n# coupling: %f\n# repetitions: %i\n", T, J, STEPS_REPEAT);
    fprintf(resf, "# simulation_t_max: %i\n# thermalization_time: %i\n# time_between_measurements: %i\n# base_random_seed: %i\n",  T_MAX_SIM, T_MEASURE_WAIT, T_MEASURE_INTERVAL, SEED);
    fprintf(resf, "# extra:\n# area: %i\n# active_spins_excluding_boundaries:%i\n# total_independent_sims: %i\n", AREA, NTOT, MULTISIZE*STEPS_REPEAT);

    // device grid
    MULTISPIN * dev_grid;
    cudaMalloc(&dev_grid, L*L*sizeof(MULTISPIN));

    float n_measures_per_sim = (float) ((T_MAX_SIM - T_MEASURE_WAIT)/T_MEASURE_INTERVAL);

    //OUTER REP LOOP  
    struct multiavg_tr single_run_avgs = new_multiavg_tr(n_measures_per_sim);
    struct multiavg_tr * dev_single_run_avgs;
    cudaMalloc(&dev_single_run_avgs, sizeof(struct multiavg_tr));
    cudaMemcpy(dev_single_run_avgs, &single_run_avgs, sizeof(struct multiavg_tr), cudaMemcpyHostToDevice);

    float * dev_partial_res;
    cudaMalloc(&dev_partial_res, sizeof(float));

    for( int krep=0; krep< STEPS_REPEAT; krep++) {
        //srand(SEED + krep);
        initRNG<<<1, NTOT>>>(rngStates, SEED+krep);

        cudaMemcpy(dev_grid, startgrid, L*L*sizeof(MULTISPIN), cudaMemcpyHostToDevice);

        // INNER SIM LOOPS
        printf("# simulation %i\n", krep+1);
        printf("#    waiting thermalization for the first %i sim steps.\n", T_MEASURE_WAIT);
        int ksim=0;
        for( ; ksim<T_MEASURE_WAIT; ksim++) {
            dev_update_grid<<<1,NTOT>>>(dev_grid, exp4, exp8, rngStates );
        }
        printf("#    finished thermalization. running %i more simulation steps and performing %f measures.\n",(T_MAX_SIM - T_MEASURE_WAIT), n_measures_per_sim);

        for( ; ksim<T_MAX_SIM; ksim++) {
            dev_update_grid<<<1,NTOT>>>(dev_grid, exp4, exp8, rngStates );
            
            if( ksim % T_MEASURE_INTERVAL == 0) {
                dev_update_magnetization_tracker<<<1,NTOT>>>(dev_grid, dev_single_run_avgs, dev_partial_res, krep);                
            }
        }
        // END INNER SIM LOOPS        
        printf("# end simulation %i\n", krep+1);
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

    multidump_a_few(endgrid);


}



int main() {
    // still used for init_random_grid
    srand(SEED);

    // curand init
    // Allocate memory for RNG states
    curandState *d_rngStates = 0;
    // cudaMalloc((void **)&d_rngStates, grid.x * block.x * sizeof(curandState));
    cudaMalloc((void **)&d_rngStates, NTOT*sizeof(curandState));
    // Initialise RNG
    initRNG<<<1, NTOT>>>(d_rngStates, SEED);
     
    // as far as I understand, the exponentials cannot be calculated in the global scope because they don't qualify
    // as constant expressions, so they have to be calculated here and propagated all the way.
    const float EXP4 = exp( -(4.*J) / T);
    const float EXP8 = exp( -(8.*J) / T);

    // original grid on the cpu
    MULTISPIN startgrid[L*L];
    init_random_grid(startgrid);
    // multidump_a_few(startgrid);

    parall_measure_cycle(startgrid, EXP4, EXP8, d_rngStates);

    //cudaFree(d_rngStates);

    return 0;
}

