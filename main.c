#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "utils.h"
#include "fdiff.h"

#define GRID_SIZE_X 9
#define GRID_SIZE_Y 32

int main(int argc, char** argv){
    int myid, nprocs;
    int nx, ny;
    Grid u;
    MPI_Topology top;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    
    if(nprocs!=8){
        if(myid==0)
            printf("Only functons for 8 processors.\n");
        MPI_Finalize();
        return 1;
    }

    nx = GRID_SIZE_X;
    ny = GRID_SIZE_Y;
    
    top = init_top(nprocs, myid, MPI_COMM_WORLD);
    
    u = init_grid(nx/3, ny/4);
    
    init_boundaries(u, top);
    
    if(myid==0)
        printf("\nINITIALISED GRID\n\n");    
    MPI_Barrier(MPI_COMM_WORLD);
    print_grid(u, top);   
    
    precond_CG(u, top);
    
    if(myid==0)
        printf("\nSOLVED GRID\n\n");    

    MPI_Barrier(MPI_COMM_WORLD);
    print_grid(u, top);   
     
    free_grid(&u);

    MPI_Finalize();
    
    return 0;
}
