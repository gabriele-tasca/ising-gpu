#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

// hard parameters
#define L 150
#define MULTISPIN unsigned int
#define MULTISIZE 32
// #define MULTISPIN unsigned long int
// #define MULTISIZE 64

#define STEPS_REPEAT 3

const int AREA = L*L;
const int NTOT = (L-2)*(L-2);


//external parameters

struct parameters {
    float t;
    float t_start;
    float j;
    int steps_repeat;
    int t_max_sim;
    int t_measure_wait;
    int t_measure_interval;
    int seed;
};



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


int main() {
    // read params

    FILE *param_f = fopen("params.txt", "r");
    struct parameters par;

    fscanf(param_f, "temperature %f\ntemperature_start %f\ncoupling %f\nsimulation_t_max %i\nthermalization_time %i\ntime_between_measurements %i\nbase_random_seed %i\n", 
                    &(par.t),     &(par.t_start),     &(par.j),    &(par.t_max_sim),   &(par.t_measure_wait),   &(par.t_measure_interval),       &(par.seed));
    printf("%f\n", par.t);
    printf("%f\n", par.t_start);
    printf("%f\n", par.j);
    printf("%i\n", par.t_max_sim);
    printf("%i\n", par.t_measure_wait);
    printf("%i\n", par.t_measure_interval);
    printf("%i\n", par.seed);
    fclose(param_f);


    srand(par.seed);
    MULTISPIN startgrid[L*L];
    init_t0_grid(startgrid);



    const double EXP4 = exp( -(4.*par.j) / par.t);
    const double EXP8 = exp( -(8.*par.j) / par.t);

    // measure cycle
    FILE *resf = fopen("results.txt", "w");
    fprintf(resf, "# cpu2\n");
    fprintf(resf, "# hard-coded parameters:\n# linear_size: %i\n# spin_coding_size: %i\n", L, MULTISIZE);
    fprintf(resf, "# parameters:\n# temperature: %f\n#temp_start: %f\n# coupling: %f\n# repetitions: %i\n", par.t, par.t_start, par.j, STEPS_REPEAT);
    fprintf(resf, "# simulation_t_max: %i\n# thermalization_time: %i\n# time_between_measurements: %i\n# base_random_seed: %i\n",  par.t_max_sim, par.t_measure_wait, par.t_measure_interval, par.seed);
    fprintf(resf, "# extra:\n# area: %i\n# active_spins_excluding_boundaries:%i\n# total_independent_sims: %i\n", AREA, NTOT, MULTISIZE*STEPS_REPEAT);
    
    MULTISPIN grid[L*L];
    double n_measures_per_sim = (double) ((par.t_max_sim - par.t_measure_wait)/par.t_measure_interval);

    //OUTER REP LOOP  
    struct multiavg_tr single_run_avgs = new_multiavg_tr(n_measures_per_sim);

    for( int krep=0; krep< STEPS_REPEAT; krep++) {
        srand(par.seed + krep);
        memcpy(grid, startgrid, L*L*sizeof(MULTISPIN) );

        // INNER SIM LOOPS
        printf("# simulation %i\n", krep+1);
        printf("#    waiting thermalization for the first %i sim steps.\n", par.t_measure_wait);
        int ksim=0;
        for( ; ksim<par.t_measure_wait; ksim++) {
            update_grid_black(grid, EXP4, EXP8);
            update_grid_white(grid, EXP4, EXP8);
        }
        printf("#    finished thermalization. running %i more simulation steps and performing %f measures.\n",(par.t_max_sim - par.t_measure_wait), n_measures_per_sim);

        for( ; ksim<par.t_max_sim; ksim++) {
            update_grid_black(grid, EXP4, EXP8);
            update_grid_white(grid, EXP4, EXP8);
            
            if( ksim % par.t_measure_interval == 0) {
                update_magnetization_tracker(&single_run_avgs, grid, krep);
            }
        }
        // END INNER SIM LOOPS        
        printf("# end simulation %i\n", krep+1);
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





    return 0;
}

