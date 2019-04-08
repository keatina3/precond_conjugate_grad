#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "fdiff.h"
#include "utils.h"

#define GRID_SIZE_X 12
#define GRID_SIZE_Y 9
#define ERR 1.0E-07

int main(int argc, char** argv){
    int myid, nprocs;
    int n,m;
    int nx, ny;
    double **grid, *grid_vals;
    int count;
    MPI_Datatype column;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    
    /*
    int dims[2] = {4,3};
    int periods[2] = {0,0};

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);
    */
    
    n = 3, m = 4;
    nx = GRID_SIZE_X+2;
    ny = GRID_SIZE_Y+2;

    //init_grid(grid, grid_vals, nx, ny);
    //init_boundaries();


    MPI_Finalize();
    
    return 0;
}
