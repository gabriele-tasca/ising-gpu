#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define L 198
static int AREA = L*L;
static int NTOT = L*L - (4*L -4);

// #define T 6.
// #define T 0.1
// #define T 2.26918531421
#define T_CYCLE_START 1.6
#define T_CYCLE_END 2.9
#define T_CYCLE_STEP 0.1

#define SINGLETEMP 3.0

int n_temps = ( T_CYCLE_END - T_CYCLE_START )/ (T_CYCLE_STEP);


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
    .steps_repeat = 64,
    .t_max_sim = 200,
    .t_measure_wait = 50,
    .t_measure_interval = 10  };


// average tracker struct 
struct avg_tr {
    double sum;
    double sum_squares;
    int n;
};
struct avg_tr new_avg_tr(int locn) {
    struct avg_tr a = { .sum = 0, .sum_squares = 0, .n = locn};
    return a;
}
void update_avg(struct avg_tr * tr_p, double newval) {
    tr_p->sum +=  newval;
    tr_p->sum_squares += (newval*newval);
}
double average( struct avg_tr tr) {
    return (tr.sum)/((double) tr.n) ;
}
double stdev( struct avg_tr tr) {
    return sqrt(  ( tr.sum_squares)/((double) tr.n)  -  pow(( (tr.sum)/((double) tr.n) ),2)  );
}
double variance( struct avg_tr tr) {
    return (  ( tr.sum_squares)/((double) tr.n)  -  pow(( (tr.sum)/((double) tr.n) ),2)  );
}


double unitrand(){
    return (double)rand() / (double)RAND_MAX;
}

void init_random(char grid[L][L]) {
    for(int x = 0; x<L; x++) {
        for(int y = 0; y<L; y++) {
            grid[x][y] = rand() & 1;
        }
    }
}
void init_t0(char grid[L][L]) {
    for(int x = 0; x<L; x++) { 
        for(int y = 0; y<L; y++) {
            grid[x][y] = 0;
        }
    }
}


// can segfault 
char grid_step(char grid[L][L], int x, int y, int xstep, int ystep) {
    return grid[x+xstep][y+ystep];
}

// segfault if applied to an edge spin, must be called only on the inner L-1 grid
// *2 -4 remaps {0,1} into {-1,1}
char deltaH(char grid[L][L], int x, int y) {
    char s0 = grid[x][y];
    char j1 = s0 ^ grid_step(grid, x, y, 1, 0);
    char j2 = s0 ^ grid_step(grid, x, y, -1, 0);
    char j3 = s0 ^ grid_step(grid, x, y, 0, 1);
    char j4 = s0 ^ grid_step(grid, x, y, 0, -1);
    return -((j1 + j2 + j3 + j4) *2 -4)*2*J;
}

void flip(char grid[L][L], int x, int y) {
    grid[x][y] = !grid[x][y];
}

void update_spin(char grid[L][L], int x, int y, double temperature) {
    double dh = (double) deltaH(grid, x, y);
    // printf("dh: %f \n", dh);

    double p = exp(  -dh / temperature);
    double ur = unitrand(); //CHANGE
    // printf("p: %f, unitrand: %f \n", p, ur);
    if(ur < p ) {
        flip(grid, x, y);
    } 
}

void update_grid_white(char grid[L][L], double temperature) {
    for(int x = 1; x<L-1; x+=1) {
        for(int y = (1 + x%2) ; y<L-1; y+=2) {
            update_spin(grid, x, y, temperature);
        }
    }
}
void update_grid_black(char grid[L][L], double temperature) {
    for(int x = 1; x<L-1; x+=1) {
        for(int y = (1 + (x+1)%2) ; y<L-1; y+=2) {
            update_spin(grid, x, y, temperature);
        }
    }
}

void dump(char grid[L][L]) {
    for(int x = 0; x<L; x++) {
        for(int y = 0; y<L; y++) {
            // if(grid[x][y] == 0) printf("•");
            // else printf("◘");
            if(grid[x][y] == 0) printf(" ");
            else printf("█");
            // printf("%i", grid[x][y]);
        }
        printf("\n");
    }
    printf("\n");
}

double measure_m(char grid[L][L]) {
    int m = 0;
    for(int x = 1; x<L-1; x++) {
        for(int y = 1; y<L-1; y++) {
            m += (grid[x][y]*2. -1.);
            // printf("x %i m %f \n", x, grid[x][y] );
        }
    }
    return (((double) m ) / (double) NTOT) ;
}

void measure_cycle(char startgrid[L][L], struct measure_plan pl, FILE *resf, double temperature) {
    char grid[L][L];


 
    //OUTER REP LOOP
    double n_measures_per_sim = (double) ((pl.t_max_sim - pl.t_measure_wait)/pl.t_measure_interval);
    struct avg_tr avg_of_all_sims_tr = new_avg_tr(pl.steps_repeat);

    float avg_of_sims = 0;
    for( int krep=0; krep< pl.steps_repeat; krep++) {
        
        srand(SEED + krep);

        memcpy(grid, startgrid, L*L*sizeof(char) );
    

        // INNER SIM LOOPS
        if(HISTORY) printf("# simulation %i\n", krep+1);
        if(HISTORY) printf("#    waiting thermalization for the first %i sim steps\n", pl.t_measure_wait);
        int ksim=0;
        for( ; ksim<pl.t_measure_wait; ksim++) {
            update_grid_black(grid, temperature);
            update_grid_white(grid, temperature);
            if( ksim % pl.t_measure_interval == 0) {
                // print all history
                if(HISTORY) printf("%i %f\n", ksim, measure_m(grid));
            }

        }
        if(HISTORY) printf("#    end thermalization\n");

        struct avg_tr sim_avg_tr = new_avg_tr(n_measures_per_sim);
        for( ; ksim<pl.t_max_sim; ksim++) {
            update_grid_black(grid, temperature);
            update_grid_white(grid, temperature);
            
            if( ksim % pl.t_measure_interval == 0) {
                double locres = measure_m(grid);
                // print all history
                if(HISTORY) printf("%i %f\n", ksim, locres);
                update_avg(&sim_avg_tr, locres);
            }

        }
        // END INNER SIM LOOPS        
        if(HISTORY) printf("# end simulation %i\n", krep+1);
        if(HISTORY) printf("# average for simulation %i: %f +- %f \n", krep+1, average(sim_avg_tr), stdev(sim_avg_tr));
        update_avg(&avg_of_all_sims_tr, average(sim_avg_tr));
    }
    // END OUTER REP LOOP

    fprintf(resf, "%f ", temperature);
    fprintf(resf, "%f ", average(avg_of_all_sims_tr));
    fprintf(resf, "%f\n", stdev(avg_of_all_sims_tr));
    // fprintf(resf, "\n\n");

    if(HISTORY) dump(grid);
}



int main() {

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
    char startgrid[L][L];
    init_t0(startgrid);

    // dump(startgrid);

    // cycle
    for( double kt=T_CYCLE_START; kt<T_CYCLE_END; kt+=T_CYCLE_STEP ) {
        measure_cycle(startgrid, PLAN, resf, kt);
    }

    // just one
    // measure_cycle(startgrid, PLAN, resf, SINGLETEMP);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float total_time = 0;
    cudaEventElapsedTime(&total_time, start, stop);

    FILE *timef = fopen("time.txt", "w");
    long int total_flips = ((long int)(n_temps))* ((long int)((PLAN.steps_repeat))) * ((long int)(PLAN.t_max_sim)) * ((long int)(NTOT));
    
    fprintf(timef, "# cpu1\n");
    fprintf(timef, "# total execution time (milliseconds):\n");
    fprintf(timef, "%f\n", total_time);
    fprintf(timef, "# total spin flips performed:\n");
    fprintf(timef, "%li\n", total_flips);
    fprintf(timef, "# average spin flips per millisecond:\n");
    fprintf(timef, "%Lf\n", ((long double) total_flips  )/( (long double) total_time ) );

    fclose(timef);

    fclose(resf);
}

