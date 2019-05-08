#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <unistd.h>
#include "utils.h"

// initialises custom topology for 8 procs in shape of grid //
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

// decomposes length n array in size near equal parts //
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

// intialises grid structure //
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
    double sum = 0.0;
    int i,j;

    for(i=1;i<=x.dims[0];i++)
        for(j=1;j<=x.dims[1];j++)
            sum += x.arr[i][j]*x.arr[i][j];

    return sum;
}

// inner product //
double frob_inner_prod(Grid x, Grid y){
    int i,j;
    double sum = 0.0;

    for(i=1;i<=x.dims[0];i++)
        for(j=1;j<=x.dims[1];j++)
            sum += x.arr[i][j]*y.arr[i][j];
    return sum;
}

// vector scalar product //
void vec_scal_prod(Grid ax, Grid x, double a){
    int i,j;

    for(i=1;i<=x.dims[0];i++)
        for(j=1;j<=x.dims[1];j++)
            ax.arr[i][j] = a*x.arr[i][j];
}

// vector vector addition //
void vec_vec_add(Grid x_y, Grid x, Grid y){
    int i,j;

    for(i=1;i<=x.dims[0];i++)
        for(j=1;j<=x.dims[1];j++)
            x_y.arr[i][j] = x.arr[i][j] + y.arr[i][j];
}

void print_grid(Grid x, MPI_Topology top){
    int i,j;

    for(i=0;i<3;i++){
        if(top.coords[0]==i && top.coords[1]==3){
            if(i==0)
                printf("%lf ", x.arr[0][x.dims[1]+1]);
            for(j=1;j<=x.dims[0];j++)
                printf("%lf ", x.arr[j][x.dims[1]+1]);
            if(i==2)
                printf("%lf\n", x.arr[x.dims[0]+1][x.dims[1]+1]);
        }
        fflush(stdout);
        usleep(1000);
        MPI_Barrier(top.comm);
    }
    
    for(j=x.dims[1];j>0;j--){
        print_col(x, j, 3, 3, top);
        MPI_Barrier(top.comm); 
    }
    
    if(top.coords[0]==0 && top.coords[1]==2){
        for(j=0;j<=x.dims[0]+1;j++)
            printf("%lf ", x.arr[j][x.dims[1]]);
    }
    fflush(stdout);
    usleep(1000);
    MPI_Barrier(top.comm);

    for(i=1;i<3;i++){
        if(top.coords[0]==i && top.coords[1]==3){
            if(i==2)
                printf("%lf ", x.arr[1][x.dims[1]+1]);
            for(j=2;j<=x.dims[0];j++)
                printf("%lf ", x.arr[j][x.dims[1]+1]);
            if(i==2)
                printf("%lf\n", x.arr[x.dims[0]+1][x.dims[1]+1]);
        }
        fflush(stdout);
        usleep(1000);
        MPI_Barrier(top.comm);
    }
    
    
    for(j=x.dims[1]-1;j>1;j--){
        print_col(x, j, 2, 1, top); 
        MPI_Barrier(top.comm); 
    }
    
    if(top.coords[0]==0 && top.coords[1]==2){
        for(j=0;j<=x.dims[0]+1;j++)
            printf("%lf ", x.arr[j][1]);
    }
    fflush(stdout);
    usleep(1000);
    MPI_Barrier(top.comm);
    
    if(top.coords[0]==1 && top.coords[1]==3){
        for(j=1;j<=x.dims[0];j++)
            printf("%lf ", x.arr[j][x.dims[1]+1]);
        printf("\n");
    }
    fflush(stdout);
    usleep(1000);
    MPI_Barrier(top.comm);

    for(i=1;i>=0;i--){
        for(j=x.dims[1];j>0;j--){
            print_col(x, j, i, 2, top);
            MPI_Barrier(top.comm); 
        }
    }

    for(i=0;i<2;i++){
        if(top.coords[0]==i && top.coords[1]==0){
            if(i==0)
                printf("%lf ", x.arr[0][0]);
            for(j=1;j<=x.dims[0];j++)
                printf("%lf ", x.arr[j][0]);
            if(i==1)
                printf("%lf\n", x.arr[x.dims[0]+1][0]);
        }
        fflush(stdout);
        usleep(1000);
        MPI_Barrier(top.comm);
    }
}

void print_col(Grid x, int col, int yproc, int xprocs, MPI_Topology top){
    int i,j;

    for(i=0;i<xprocs;i++){
        if(top.coords[0]==i && top.coords[1]==yproc){
            if(i==0)
                printf("%lf ", x.arr[0][col]);
            for(j=1;j<=x.dims[0];j++)
                printf("%lf ", x.arr[j][col]);
            if(i==(xprocs-1))
                printf("%lf\n", x.arr[x.dims[0]+1][col]);
        }
        fflush(stdout);
        usleep(1000);
        MPI_Barrier(top.comm);
    }
}
