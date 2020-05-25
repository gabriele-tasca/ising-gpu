#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define L 500
static int AREA = L*L;
static int NTOT = L*L - (4*L -4);

// #define T 6.
// #define T 0.1
// #define T 2.26918531421
#define T 2.24

#define J 1.

#define SEED 1000

struct measure_plan {
    int steps_repeat;
    int t_max_sim;
    int t_measure_wait;
    int t_measure_interval; } 
static PLAN = {
    .steps_repeat = 50,
    .t_max_sim = 600,
    .t_measure_wait = 100,
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

void init_random(short int grid[L][L]) {
    for(int x = 0; x<L; x++) {
        for(int y = 0; y<L; y++) {
            grid[x][y] = rand() & 1;
        }
    }
}
void init_t0(short int grid[L][L]) {
    for(int x = 0; x<L; x++) {
        for(int y = 0; y<L; y++) {
            grid[x][y] = 0;
        }
    }
}


// can segfault 
short int grid_step(short int grid[L][L], int x, int y, int xstep, int ystep) {
    return grid[x+xstep][y+ystep];
}

// segfault if applied to an edge spin, must be called only on the inner L-1 grid
// *2 -4 remaps {0,1} into {-1,1}
short int deltaH(short int grid[L][L], int x, int y) {
    short int s0 = grid[x][y];
    short int j1 = s0 ^ grid_step(grid, x, y, 1, 0);
    short int j2 = s0 ^ grid_step(grid, x, y, -1, 0);
    short int j3 = s0 ^ grid_step(grid, x, y, 0, 1);
    short int j4 = s0 ^ grid_step(grid, x, y, 0, -1);
    return -((j1 + j2 + j3 + j4) *2 -4)*2*J;
}

void flip(short int grid[L][L], int x, int y) {
    grid[x][y] = !grid[x][y];
}

void update_spin(short int grid[L][L], int x, int y) {
    double dh = (double) deltaH(grid, x, y);
    // printf("dh: %f \n", dh);

    double p = exp(  -dh / T);
    double ur = unitrand(); //CHANGE
    // printf("p: %f, unitrand: %f \n", p, ur);
    if(ur < p ) {
        flip(grid, x, y);
    } 
}

void update_grid_white(short int grid[L][L]) {
    for(int x = 1; x<L-1; x+=1) {
        for(int y = (1 + x%2) ; y<L-1; y+=2) {
            update_spin(grid, x, y);
        }
    }
}
void update_grid_black(short int grid[L][L]) {
    for(int x = 1; x<L-1; x+=1) {
        for(int y = (1 + (x+1)%2) ; y<L-1; y+=2) {
            update_spin(grid, x, y);
        }
    }
}

void dump(short int grid[L][L]) {
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

double measure_m(short int grid[L][L]) {
    int m = 0;
    for(int x = 1; x<L-1; x++) {
        for(int y = 1; y<L-1; y++) {
            m += (grid[x][y]*2. -1.);
            // printf("x %i m %f \n", x, grid[x][y] );
        }
    }
    return (((double) m ) / (double) NTOT) ;
}

void measure_cycle(short int startgrid[L][L], struct measure_plan pl) {
    FILE *resf = fopen("results.txt", "w");
    short int grid[L][L];
    fprintf(resf, "# cpu1\n");
    fprintf(resf, "# parameters:\n# linear_size: %i\n", L);
    fprintf(resf, "# temperature: %f\n#temp_start: %f\n# coupling: %f\n# repetitions: %i\n", T, 0., J, pl.steps_repeat);
    fprintf(resf, "# simulation_t_max: %i\n# thermalization_time: %i\n# time_between_measurements: %i\n# base_random_seed: %i\n",  pl.t_max_sim, pl.t_measure_wait, pl.t_measure_interval, SEED);
    fprintf(resf, "# extra:\n# area: %i\n# active_spins_excluding_boundaries:%i\n", AREA, NTOT);
  

    fprintf(resf, "# columns: MC time -- magnetization\n");

    //OUTER REP LOOP
    double n_measures_per_sim = (double) ((pl.t_max_sim - pl.t_measure_wait)/pl.t_measure_interval);
    struct avg_tr overall_avg_tr = new_avg_tr(n_measures_per_sim*pl.steps_repeat);
    struct avg_tr avg_of_all_sims_tr = new_avg_tr(pl.steps_repeat);

    float avg_of_sims = 0;
    for( int krep=0; krep< pl.steps_repeat; krep++) {
        
        srand(SEED + krep);

        memcpy(grid, startgrid, L*L*sizeof(short int) );
    

        // INNER SIM LOOPS
        fprintf(resf, "# simulation %i\n", krep+1);
        fprintf(resf, "#    waiting thermalization for the first %i sim steps\n", pl.t_measure_wait);
        int ksim=0;
        for( ; ksim<pl.t_measure_wait; ksim++) {
            update_grid_black(grid);
            update_grid_white(grid);
            if( ksim % pl.t_measure_interval == 0) {
                fprintf(resf, "%i %f\n", ksim, measure_m(grid));
            }

        }
        fprintf(resf, "#    end thermalization\n");

        struct avg_tr sim_avg_tr = new_avg_tr(n_measures_per_sim);
        for( ; ksim<pl.t_max_sim; ksim++) {
            update_grid_black(grid);
            update_grid_white(grid);
            
            if( ksim % pl.t_measure_interval == 0) {
                double locres = measure_m(grid);
                fprintf(resf, "%i %f\n", ksim, locres);
                update_avg(&overall_avg_tr, locres);
                update_avg(&sim_avg_tr, locres);
            }

        }
        // END INNER SIM LOOPS        
        fprintf(resf, "# end simulation %i\n", krep+1);
        fprintf(resf, "# average for simulation %i: %f +- %f \n", krep+1, average(sim_avg_tr), stdev(sim_avg_tr));
        update_avg(&avg_of_all_sims_tr, average(sim_avg_tr));
    }
    // END OUTER REP LOOP

    fprintf(resf, "# average of all simulations: %f +- %f\n", average(avg_of_all_sims_tr), stdev(avg_of_all_sims_tr));
    fprintf(resf, "# overall average: %f +- %f\n", average(overall_avg_tr), stdev(overall_avg_tr));
    
    dump(grid);
}



int main() {
    srand(SEED);
    short int startgrid[L][L];
    init_t0(startgrid);

    // dump(startgrid);



    measure_cycle(startgrid, PLAN);

}

