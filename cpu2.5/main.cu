#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

// hard parameters
#define L 198
#define MULTISPIN unsigned int
#define MULTISIZE 32
// #define MULTISPIN unsigned long int
// #define MULTISIZE 64



const int AREA = L*L;
const int NTOT = (L-2)*(L-2);

#define T_CYCLE_START 1.6
#define T_CYCLE_END 2.9
#define T_CYCLE_STEP 0.1

#define SINGLETEMP 3.0

int n_temps = ( T_CYCLE_END - T_CYCLE_START )/ (T_CYCLE_STEP);


#define STEPS_REPEAT 3

#define J 1.

#define SEED 1000

// print history true/false
#define HISTORY 1

struct measure_plan {
    int steps_repeat;
    int t_max_sim;
    int t_measure_wait;
    int t_measure_interval; } 
static PLAN = {
    .steps_repeat = 2,
    .t_max_sim = 200,
    .t_measure_wait = 50,
    .t_measure_interval = 10  };



// average tracker struct
struct avg_tr {
    double sum;
    double sum_squares;
    int n;
};
static inline struct avg_tr new_avg_tr(int locn) {
    struct avg_tr a = { .sum = 0, .sum_squares = 0, .n = locn};
    return a;
}
// if the numbers overflow, then it would be necessary to divide by N before summing
// however it's faster the other way
static inline void update_avg(struct avg_tr * tr_p, double newval) {
    tr_p->sum +=  newval;
    tr_p->sum_squares += (newval*newval);
}
static inline double average( struct avg_tr tr) {
    return (tr.sum)/((double) tr.n) ;
}
static inline double stdev( struct avg_tr tr) {
    return sqrt(  ( tr.sum_squares)/((double) tr.n)  -  pow(( (tr.sum)/((double) tr.n) ),2)  );
}
// static inline double variance( struct avg_tr tr) {
//     return (  ( tr.sum_squares)/((double) tr.n)  -  pow(( (tr.sum)/((double) tr.n) ),2)  );
// }

//multispin averages, hard-coded to track a number MULTISPIN * STEPS_REPEAT of values
struct multiavg_tr {
    double sum[MULTISIZE * STEPS_REPEAT];
    double sum_squares[MULTISIZE * STEPS_REPEAT];
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
static inline void update_multiavg(struct multiavg_tr * tr_p, double newval, int k) {
    tr_p->sum[k] +=  newval;
    tr_p->sum_squares[k] += (newval*newval);
}
static inline double multiaverage( struct multiavg_tr tr, int k) {
    return (tr.sum[k])/((double) tr.n) ;
}
static inline double multistdev( struct multiavg_tr tr, int k) {
    return sqrt(  ( tr.sum_squares[k])/((double) tr.n)  -  pow(( (tr.sum[k])/((double) tr.n) ),2)  );
}
// static inline double multivariance( struct multiavg_tr tr, int k) {
//     return (  ( tr.sum_squares[k])/((double) tr.n)  -  pow(( (tr.sum[k])/((double) tr.n) ),2)  );
// }




double unitrand(){
    return (double)rand() / (double)RAND_MAX;
}

// index has to be less that MULTISIZE
static inline void set_spin_1 (MULTISPIN * multi, int index) {
    *multi |= 1 << index;
}
static inline void set_spin_0 (MULTISPIN * multi, int index) {
    *multi &= ~(1 << index);
}
static inline MULTISPIN read_spin(MULTISPIN multi, int index) {
    return ((multi >> index) & 1);
}

// each bit exp8 and exp8 describes the Metropolis RNG result for that bit,
// specifying if the random r is bigger or smaller than the relevant values e^(4J/kT) and e^(8J/kT) (passed from outside)
static inline MULTISPIN generate_exp4_mask(double exp4, double exp8, double random_number) {
    MULTISPIN res;
    for(int k=0; k<MULTISIZE; k++) {
        if( exp4 > random_number && random_number > exp8) { // this is taken from the article and works. the version below might not but simplifies some things
        // if( exp4 > random_number) {
            set_spin_1(&res, k);
        } else {
            set_spin_0(&res, k);
        }
    }
    return res;
}
static inline MULTISPIN generate_exp8_mask(double exp8, double random_number) {
    MULTISPIN res;
    for(int k=0; k<MULTISIZE; k++) {
        if( random_number < exp8 ) {
            set_spin_1(&res, k);
        } else {
            set_spin_0(&res, k);
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
static inline MULTISPIN grid_step(MULTISPIN grid[L*L], int x, int y, int xstep, int ystep) {
    return grid[(x+xstep)  + (y+ystep)*L];
}


// segfault if applied to an edge spin, must be called only on the inner L-1 grid
void update_multispin(MULTISPIN grid[L*L], int x, int y, double exp4, double exp8 ) {

    MULTISPIN s0 = grid[x+y*L];
    
    double ur = unitrand(); 
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

// checkerboard patterns
void update_grid_white(MULTISPIN grid[L*L], double exp4, double exp8 ) {
    for(int x = 1; x<L-1; x+=1) {
        for(int y = (1 + x%2) ; y<L-1; y+=2) {
            update_multispin(grid, x, y, exp4, exp8);
        }
    }
}
void update_grid_black(MULTISPIN grid[L*L], double exp4, double exp8 ) {
    for(int x = 1; x<L-1; x+=1) {
        for(int y = (1 + (x+1)%2) ; y<L-1; y+=2) {
            update_multispin(grid, x, y, exp4, exp8);
        }
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

// non GPU
double measure_1bit_magnetization(MULTISPIN grid[L*L], int multi_index) {
    double lmagn = 0;
    for(int x = 1; x<L-1; x++) {
        for(int y = 1; y<L-1; y++) {
            double bit = (double) read_spin(grid[x+y*L], multi_index); 
            lmagn = lmagn + ((bit*2.) -1.);
        }
    }

    return (((double) lmagn ) / (double) NTOT) ;
}

// as usual, the number of independent measures is hardcoded to MULTISIZE * STEPS_REPEAT.
// this function measures a single multispin, updating a chunk of the tracker. (for example the first half if rep_steps is 2)
void update_magnetization_tracker( struct multiavg_tr * tr_p, MULTISPIN grid[L*L], int block_count) {
    for( int k=0; k < MULTISIZE; k++) {
        double mag = measure_1bit_magnetization(grid, k);
        update_multiavg(tr_p, mag, k + MULTISIZE*block_count );
    }
}

// static inline void update_multiavg(struct multiavg_tr * tr_p, double newval, int k) {
//     tr_p->sum[k] +=  newval;
//     tr_p->sum_squares[k] += (newval*newval);
// }

void measure_cycle(MULTISPIN startgrid[L*L], struct measure_plan pl, FILE *resf, double EXP4, double EXP8) {


    MULTISPIN grid[L*L];
    double n_measures_per_sim = (double) ((pl.t_max_sim - pl.t_measure_wait)/pl.t_measure_interval);

    //OUTER REP LOOP  
    struct multiavg_tr single_run_avgs = new_multiavg_tr(n_measures_per_sim);

    for( int krep=0; krep< STEPS_REPEAT; krep++) {
        srand(SEED + krep);
        memcpy(grid, startgrid, L*L*sizeof(MULTISPIN) );

        // INNER SIM LOOPS
        if(HISTORY) printf("# simulation %i\n", krep+1);
        if(HISTORY) printf("#    waiting thermalization for the first %i sim steps.\n", pl.t_measure_wait);
        int ksim=0;
        for( ; ksim<pl.t_measure_wait; ksim++) {
            update_grid_black(grid, EXP4, EXP8);
            update_grid_white(grid, EXP4, EXP8);
        }
        if(HISTORY) printf("#    finished thermalization. running %i more simulation steps and performing %f measures.\n",(pl.t_max_sim - pl.t_measure_wait), n_measures_per_sim);

        for( ; ksim<pl.t_max_sim; ksim++) {
            update_grid_black(grid, EXP4, EXP8);
            update_grid_white(grid, EXP4, EXP8);
            
            if( ksim % pl.t_measure_interval == 0) {
                update_magnetization_tracker(&single_run_avgs, grid, krep);
            }
        }
        // END INNER SIM LOOPS        
        if(HISTORY) printf("# end simulation %i\n", krep+1);
    }
    // END OUTER REPETITION LOOP

    struct avg_tr avg_of_blocks = new_avg_tr( MULTISIZE * STEPS_REPEAT );

    for(int k=0; k < MULTISIZE * STEPS_REPEAT; k++) {
        double lres = multiaverage(single_run_avgs, k);
        double lstdev = multistdev(single_run_avgs, k);

        fprintf(resf, "# average of simulation %i\n: %f +- %f\n", k+1, lres, lstdev);
        update_avg(&avg_of_blocks, lres);
    }
    fprintf(resf, "# overall average \n: %f +- %f\n", average(avg_of_blocks), stdev(avg_of_blocks));

    multidump_a_few(grid);

}


int main() {
    // read params


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);


    FILE *resf = fopen("results.txt", "w");
    fprintf(resf, "# cpu1\n");
    fprintf(resf, "# parameters:\n# linear_size: %i\n", L);
    fprintf(resf, "#temp_start: %f\n# coupling: %f\n# repetitions: %i\n", 0., J, PLAN.steps_repeat);
    fprintf(resf, "# simulation_t_max: %i\n# thermalization_time: %i\n# time_between_measurements: %i\n# base_random_seed: %i\n",  PLAN.t_max_sim, PLAN.t_measure_wait, PLAN.t_measure_interval, SEED);
    fprintf(resf, "# extra:\n# area: %i\n# active_spins_excluding_boundaries:%i\n", AREA, NTOT);
    fprintf(resf, "\n");
    fprintf(resf, "# columns: temperature - average magnetization - uncertainty \n");


    srand(SEED);
    MULTISPIN startgrid[L*L];
    init_t0_grid(startgrid);







    // cycle
    for( double kt=T_CYCLE_START; kt<T_CYCLE_END; kt+=T_CYCLE_STEP ) {
        double EXP4 = exp( -(4.*J) / kt);
        double EXP8 = exp( -(8.*J) / kt);
        measure_cycle(startgrid, PLAN, resf, EXP4, EXP8);
    }


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float total_time = 0;
    cudaEventElapsedTime(&total_time, start, stop);

    FILE *timef = fopen("time.txt", "w");
    long int total_flips = ((long int)(n_temps))* ((long int)((PLAN.steps_repeat))) * ((long int)(PLAN.t_max_sim)) * ((long int)(MULTISIZE)) * ((long int)(NTOT));
    
    fprintf(timef, "# cpu1\n");
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

