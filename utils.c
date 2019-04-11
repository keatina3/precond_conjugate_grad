#include <stdlib.h>
#include <mpi.h>
#include "utils.h"

MPI_Topology init_top(int nprocs, int myid, MPI_Comm comm){
    MPI_Topology top;
    int i,j;
    int count = 0;
 
    top.comm = comm;

    top.nprocs = nprocs;
    for(j=0;j<2;j++){
        for(i=0;i<2;i++){
            if(myid==count){
                top.coords[0] = i;
                top.coords[1] = j;
                top.nbrdwn = j==0 ? -2 : count-2;
                top.nbrup = (i==1 && j ==1)  ?  -2 : count+2;
                top.nbrleft = i==0 ? -2 : count-1;
                top.nbrright = i==0 ? count+1 : -2;
            }
            count++;
        }
    }

    if(myid==count){
        top.coords[0] = 0;
        top.coords[1] = 2;
        top.nbrdwn = count-2;
        top.nbrup =  count+1;
        top.nbrleft = -2;
        top.nbrright = -2;
    }
    count++;

    for(i=0;i<3;i++){ 
        if(myid==count){
            top.coords[0] = i;
            top.coords[1] = 3;
            top.nbrdwn = i==0 ? count-1 : -2;
            top.nbrup = -2;
            top.nbrleft = i==0 ? -2 : count-1;
            top.nbrright = i==2 ? -2 : count+1;
        }
        count++;
    }

    return top;
}

int decomp1d(int n, int size, int rank, int *s, int *e){
    int nlocal, deficit;

    nlocal  = n / size;
    *s  = rank * nlocal + 1;
    deficit = n % size;
    *s  = *s + ((rank < deficit) ? rank : deficit);
    if (rank < deficit) nlocal++;
    *e      = *s + nlocal - 1;
    if (*e > n || rank == size-1) *e = n;

    return 0;
}

Grid init_grid(int n, int m){
    int i;
    Grid x;

    x.dims[0] = n;
    x.dims[1] = m;

    x.arr = (double**)malloc((n+2)*sizeof(double*));
    x.vals = (double*)calloc((n+2)*(m+2),sizeof(double));

    for(i=0;i<(n+2);i++)
        x.arr[i] = &x.vals[i*(m+2)];
    
    return x;
}

void free_grid(Grid_ptr x){
    free(x->vals);
    free(x->arr);
}

double grid_diff(Grid x){
    double sum = 0;
    int i,j;

    for(i=0;i<x.dims[0];i++)
        for(j=0;j<x.dims[1];j++)
            sum += x.arr[i][j]*x.arr[i][j];

    return sum;
}

double frob_inner_prod(Grid x, Grid y){
    int i,j;
    double sum = 0;

    for(i=1;i<x.dims[0]-1;i++)
        for(j=1;j<x.dims[1]-1;j++)
            sum += x.arr[i][j]*y.arr[i][j];
    return sum;
}

void vec_scal_prod(Grid ax, Grid x, double a){
    int i,j;

    for(i=1;i<x.dims[0]-1;i++)
        for(j=1;j<x.dims[1]-1;j++)
            ax.arr[i][j] = a*x.arr[i][j];
}

void vec_vec_add(Grid x_y, Grid x, Grid y){
    int i,j;

    for(i=1;i<x.dims[0]-1;i++)
        for(j=1;j<x.dims[1]-1;j++)
            x_y.arr[i][j] = x.arr[i][j] + y.arr[i][j];
}
