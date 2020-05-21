#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include "curand_kernel.h"

#define L 200
const int AREA = L*L;
const int NTOT = (L-2)*(L-2);

// #define T 6.
// #define T 0.1
// #define T 2.26918531421

#define T 6

#define J 1.

#define SEED 1002

struct measure_plan {
    int steps_repeat;
    int t_max_sim;
    int t_measure_wait;
    int t_measure_interval; } 
static PLAN = {
    .steps_repeat = 5,
    .t_max_sim = 100,
    .t_measure_wait = 0,
    .t_measure_interval = 2  };


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
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Initialise the RNG
    curand_init(seed, tid, 0, &rngStates[tid]);
}

float unitrand(){
    return (float)rand() / (float)RAND_MAX;
}
__device__ static inline float dev_unitrand( curandState * const rngStates, int tid ){
    curandState localState = rngStates[tid];
    return curand_uniform(&localState);
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


// can segfault 
__device__ char dev_grid_step(char grid[L*L], int x, int y, int xstep, int ystep) {
    return grid[(x+xstep)  + (y+ystep)*L];
}

// segfault if applied to an edge spin, call only on the inner L-1 grid
// *2 -4 remaps {0,1} into {-1,1}
// __device__ char dev_deltaH(char grid[L*L], int x, int y) {
//     char s0 = grid[x+y*L];
//     char j1 = s0 ^ dev_grid_step(grid, x, y, 1, 0);
//     char j2 = s0 ^ dev_grid_step(grid, x, y, -1, 0);
//     char j3 = s0 ^ dev_grid_step(grid, x, y, 0, 1);
//     char j4 = s0 ^ dev_grid_step(grid, x, y, 0, -1);
//     return -((j1 + j2 + j3 + j4) *2 -4)*2*J;
// }


// void update_spin(char grid[L*L], int x, int y) {
//     float dh = (float) deltaH(grid, x, y);
//     // printf("dh: %f \n", dh);

//     float p = exp(  -dh / T);
//     float ur = unitrand(); //CHANGE
//     // printf("p: %f, unitrand: %f \n", p, ur);
//     if(ur < p ) {
//         flip(grid, x, y);
//     }
// }
__device__ void dev_update_spin(char grid[L*L], int x, int y , curandState * const rngStates, int tid ) {
    char s0 = grid[x+y*L];
    char j1 = s0 ^ dev_grid_step(grid, x, y, 1, 0);
    char j2 = s0 ^ dev_grid_step(grid, x, y, -1, 0);
    char j3 = s0 ^ dev_grid_step(grid, x, y, 0, 1);
    char j4 = s0 ^ dev_grid_step(grid, x, y, 0, -1);
    float dh = (float) ( -((j1 + j2 + j3 + j4) *2 -4)*2*J );
    // printf("dh: %f \n", dh);

    float p = exp(  -dh / T);
    float ur = dev_unitrand(rngStates, tid); 
    // printf("p: %f, unitrand: %f \n", p, ur);
    if(ur < p ) {
        grid[x+y*L] = !grid[x+y*L];
    } 
}

// void update_grid_white(char grid[L*L]) {
//     for(int x = 1; x<L-1; x+=1) {
//         for(int y = (1 + x%2) ; y<L-1; y+=2) {
//             update_spin(grid, x, y);
//         }
//     }
// }
// void update_grid_black(char grid[L*L]) {
//     for(int x = 1; x<L-1; x+=1) {
//         for(int y = (1 + (x+1)%2) ; y<L-1; y+=2) {
//             update_spin(grid, x, y);
//         }
//     }
// }

// for now with nthreads = NTOT
__device__ void dev_update_grid(char grid[L*L], curandState * const rngStates ) {
    // assign loc_x and loc_y so that only the inner square is covered
    int loc_y = (  threadIdx.x / (L-2) ) +1;
    int loc_x = (  threadIdx.x % (L-2) ) +1;

    int tid = threadIdx.x;    // change

    // white
    if( (loc_x + loc_y%2)%2 == 0 ) {
        dev_update_spin( grid, loc_x, loc_y, rngStates, tid );
    }
    __syncthreads();
    // black
    if( (loc_x + loc_y%2)%2 == 1 ) {
        dev_update_spin( grid, loc_x, loc_y, rngStates, tid );
    }
    __syncthreads();
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

float measure_m(char grid[L*L]) {
    int m = 0;
    for(int x = 1; x<L-1; x++) {
        for(int y = 1; y<L-1; y++) {
            m += (grid[x+y*L]*2. -1.);
            // printf("x %i m %f \n", x, grid[x+y*L] );
        }
    }
    return (((float) m ) / (float) NTOT) ;
}
__device__ void dev_update_magnetization_tracker(char dev_grid[L*L], struct avg_tr * dev_tr_p, float * dev_partial_res ) {
    int y = (  threadIdx.x / (L-2) ) +1;
    int x = (  threadIdx.x % (L-2) ) +1;
        float spin = (float) dev_grid[x+y*L]; 
        atomicAdd(dev_partial_res, (spin*2.)-1.  );
        __syncthreads();
        
        if (threadIdx.x == 0) {
            *dev_partial_res = *dev_partial_res / (float) NTOT;
            dev_update_avg( dev_tr_p, *dev_partial_res);
            *dev_partial_res = 0;
        }
        __syncthreads();
    
}

__global__ void dev_measure_cycle_kernel(struct measure_plan pl, char * dev_grid, curandState * const rngStates, avg_tr * dev_single_run_avg, float * dev_partial_res  ) {
    // INNER SIM LOOPS

    int ksim=0;
    for( ; ksim<pl.t_measure_wait; ksim++) {
        dev_update_grid(dev_grid, rngStates);
    }
    // end thermalization

    for( ; ksim<pl.t_max_sim; ksim++) {
        dev_update_grid(dev_grid, rngStates);

        ////////////measures
        if( ksim % pl.t_measure_interval == 0) {
            dev_update_magnetization_tracker(dev_grid, dev_single_run_avg, dev_partial_res );
        }

    }
    // END INNER SIM LOOPS        
    ////////////measures
    // update_avg(&avg_of_all_sims_tr, average(sim_avg_tr));
}

void parall_measure_cycle(char startgrid[L*L], struct measure_plan pl, char * dev_grid, curandState * const rngStates ) {
    FILE *resf = fopen("results.txt", "w");
    fprintf(resf, "# cpu1\n");
    fprintf(resf, "# parameters:\n# linear_size: %i\n", L);
    fprintf(resf, "# temperature: %f\n#temp_start: %f\n# coupling: %f\n# repetitions: %i\n", T, 0., J, pl.steps_repeat);
    fprintf(resf, "# simulation_t_max: %i\n# thermalization_time: %i\n# time_between_measurements: %i\n# base_random_seed: %i\n",  pl.t_max_sim, pl.t_measure_wait, pl.t_measure_interval, SEED);
    fprintf(resf, "# extra:\n# area: %i\n# active_spins_excluding_boundaries:%i\n", AREA, NTOT);


    //OUTER REP LOOP
    ////////////measures
    float n_measures_per_sim = (float) ((pl.t_max_sim - pl.t_measure_wait)/pl.t_measure_interval);
    
    struct avg_tr outer_avg_tr = new_avg_tr(pl.steps_repeat);
    

    // extra space needed by dev_update_magnetization_tracker
    float * dev_partial_res;
    cudaMalloc(&dev_partial_res, sizeof(float));


    for( int krep=0; krep< pl.steps_repeat; krep++) {
        
        struct avg_tr single_run_avg = new_avg_tr(n_measures_per_sim);
        struct avg_tr * dev_single_run_avg;
        cudaMalloc(&dev_single_run_avg, sizeof(struct avg_tr));
        cudaMemcpy(dev_single_run_avg, &single_run_avg, sizeof(struct avg_tr), cudaMemcpyHostToDevice);

        initRNG<<<1, NTOT>>>(rngStates, SEED+krep);
        // initialize starting grid on the device for this sim
        cudaMemcpy(dev_grid, startgrid, L*L*sizeof(char), cudaMemcpyHostToDevice);
  
        dev_measure_cycle_kernel<<<1, NTOT>>>(pl, dev_grid, rngStates, dev_single_run_avg, dev_partial_res );

        // bring back results to CPU
        cudaMemcpy(&single_run_avg, dev_single_run_avg, sizeof(struct avg_tr), cudaMemcpyDeviceToHost);
        float lres = average(single_run_avg);
        float lstdev = stdev(single_run_avg);
        fprintf(resf, "# average of simulation %i\n: %f +- %f\n", krep+1, lres, lstdev);
        update_avg(&outer_avg_tr, lres);
    }

    // END OUTER REP LOOP
    
    ////////////measures
    fprintf(resf, "# average of all simulations: %f +- %f\n", average(outer_avg_tr), stdev(outer_avg_tr));
    
    char endgrid[L*L];
    cudaMemcpy(endgrid, dev_grid, L*L*sizeof(char), cudaMemcpyDeviceToHost);
    dump(endgrid);


}



int main() {
    srand(SEED);

    // curand init
    // Allocate memory for RNG states
    curandState *d_rngStates = 0;
    // cudaMalloc((void **)&d_rngStates, grid.x * block.x * sizeof(curandState));
    cudaMalloc((void **)&d_rngStates, NTOT*sizeof(curandState));
    // Initialise RNG
    initRNG<<<1, NTOT>>>(d_rngStates, SEED);

    // device grid
    char * dev_grid;
    cudaMalloc(&dev_grid, L*L*sizeof(char));


    char startgrid[L*L];
    init_t0(startgrid);

    dump(startgrid);



    parall_measure_cycle(startgrid, PLAN, dev_grid, d_rngStates);

    cudaFree(&d_rngStates);
    cudaFree(dev_grid);

}

