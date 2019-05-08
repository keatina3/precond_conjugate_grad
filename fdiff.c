#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "utils.h"
#include "fdiff.h"

// applying initial boundary to grid //
void init_boundaries(Grid u, MPI_Topology top){
    int i;
    if(top.coords[0]==2)
        for(i=0;i<=u.dims[1]+1;i++)
            u.arr[u.dims[0]+1][i] = 1.0;
}

// applying discrete Laplace matrix //
void laplace(Grid Lx, Grid x, MPI_Topology top){
    int i,j;
    
    for(i=0;i<(x.dims[0]+1);i++){
        for(j=0;j<(x.dims[1]+1);j++){
            Lx.arr[i][j] = -4*x.arr[i][j]; 
            if(i>0)
                Lx.arr[i][j] += x.arr[i-1][j]; 
            if(i<=x.dims[0])
                Lx.arr[i][j] += x.arr[i+1][j];
            if(j>0)
                Lx.arr[i][j] += x.arr[i][j-1];
            if(j<=x.dims[1])
                Lx.arr[i][j] += x.arr[i][j+1];
            
            // Von Neumann boundary //
            if(i==1 && top.nbrleft == -2)
                Lx.arr[i][j] += x.arr[i+1][j]; 
            if(i==x.dims[0] && top.nbrright == -2 && top.coords[1] < 3)
                Lx.arr[i][j] += x.arr[i-1][j]; 
            if(i==1 && top.nbrdwn == -2 && top.coords[1] > 0)
                Lx.arr[i][j] += x.arr[i][j+1]; 
            if(j==x.dims[1] && top.nbrup == -2)
                Lx.arr[i][j] += x.arr[i][j-1];
        }
    }
}

// calculates b-Ax //
void residual(Grid r, Grid u, MPI_Topology top){
    int i,j;
    
    for(i=1;i<=u.dims[0];i++){
        for(j=1;j<=u.dims[1];j++){
            r.arr[i][j] = 4*u.arr[i][j] - u.arr[i+1][j] - u.arr[i][j+1] 
                    - u.arr[i][j-1] - u.arr[i-1][j];
            
            // Von Neumann boundary //
            if(i==1 && top.nbrleft == -2)
                r.arr[i][j] -= u.arr[i+1][j]; 
            if(i==u.dims[0] && top.nbrright == -2 && top.coords[1] < 3)
                r.arr[i][j] -= u.arr[i-1][j]; 
            if(i==1 && top.nbrdwn == -2 && top.coords[1] > 0)
                r.arr[i][j] -= u.arr[i][j+1]; 
            if(j==u.dims[1] && top.nbrup == -2)
                r.arr[i][j] -= u.arr[i][j-1];
        }
    }
}

// applies Gauss-Seidel solver for Uij //
void GS_solve(Grid x, Grid b, MPI_Topology top, int i, int j){
    
    x.arr[i][j] = 0.25*(x.arr[i-1][j] + x.arr[i+1][j] + x.arr[i][j+1]+
                            x.arr[i][j-1] - b.arr[i][j]);
    
    // Von Neumann boundary //
    if(i==1 && top.nbrleft == -2)
        x.arr[i][j] += 0.25*x.arr[i+1][j]; 
    if(i==x.dims[0] && top.nbrright == -2 && top.coords[1] < 3)
        x.arr[i][j] += 0.25*x.arr[i-1][j]; 
    if(i==1 && top.nbrdwn == -2 && top.coords[1] > 0)
        x.arr[i][j] += 0.25*x.arr[i][j+1]; 
    if(j==x.dims[1] && top.nbrup == -2)
        x.arr[i][j] += 0.25*x.arr[i][j-1];
}    

// red-black Gauss-Seidel //
void rb_GS(Grid u, Grid b, MPI_Topology top){
    int i,j;
    int count = 0;
    Grid r;
    double glob_sum = 0, loc_sum;
    
    r = init_grid(u.dims[0], u.dims[1]);

    // datatype to send row/col of red/black grid points // 
    MPI_Datatype MPI_ROW_RED, MPI_COL_RED;
    MPI_Datatype MPI_ROW_BLACK, MPI_COL_BLACK;
     
    int odd = u.dims[1]%2 == 0 ? 0 : 1;     // if length of row/col is odd //
    MPI_Type_vector((int)(u.dims[1]+odd)/2, 1, 2, MPI_DOUBLE, &MPI_COL_RED);
    MPI_Type_vector((int)(u.dims[1])/2, 1, 2, MPI_DOUBLE, &MPI_COL_BLACK);
    MPI_Type_commit(&MPI_COL_RED);
    MPI_Type_commit(&MPI_COL_BLACK);

    odd = u.dims[0]%2 == 0 ? 0 : 1;
    MPI_Type_vector((int)(u.dims[0]+odd)/2, 1, 2*(u.dims[1]+2), MPI_DOUBLE, &MPI_ROW_RED);
    MPI_Type_vector((int)(u.dims[0])/2, 1, 2*(u.dims[1]+2), MPI_DOUBLE, &MPI_ROW_BLACK);
    MPI_Type_commit(&MPI_ROW_RED);
    MPI_Type_commit(&MPI_ROW_BLACK);
    
    // applying RB-GS algorithm //
    while(count++<MAXITERS){
        for(i=1;i<(u.dims[0]+1);i++)
            for(j=1;j<(u.dims[1]+1);j++)
                if((i+j)%2 == 0)
                    GS_solve(u, b, top, i, j);

        MPI_Barrier(top.comm);

        exchange(u, 0, top, MPI_ROW_RED, MPI_COL_RED);
    
        MPI_Barrier(top.comm);

        for(i=1;i<(u.dims[0]+1);i++)
            for(j=1;j<(u.dims[1]+1);j++)
                if((i+j)%2 == 1)
                    GS_solve(u, b, top, i, j);

        exchange(u, 1, top, MPI_ROW_BLACK, MPI_COL_BLACK);

        // checking for glob convergence //
        residual(r,u,top);
        loc_sum = grid_diff(r);
        MPI_Allreduce(&loc_sum, &glob_sum, 1, MPI_DOUBLE, MPI_SUM, top.comm); 
        if(glob_sum<ERR){
            break;
        }
    }
    free_grid(&r);
    //if(top.coords[0]==0 && top.coords[1]==0 && count==MAXITERS+1)
        //printf("Max iterations reached in G-S.\n");
}    

// halo exchange for RB-GS //
void exchange(Grid u, int odd, MPI_Topology top, MPI_Datatype MPI_ROW, MPI_Datatype MPI_COL){
        
    MPI_Sendrecv(&u.arr[u.dims[0]][odd+1], 1, MPI_ROW, top.nbrright, 0, 
                 &u.arr[0][odd+1], 1, MPI_ROW, top.nbrleft, 0, top.comm, MPI_STATUS_IGNORE);
    
    MPI_Sendrecv(&u.arr[1][odd+1], 1, MPI_ROW, top.nbrleft, 0, 
                 &u.arr[u.dims[0]+1][odd+1],1, MPI_ROW, top.nbrright,0,top.comm, MPI_STATUS_IGNORE);
    
    MPI_Sendrecv(&u.arr[odd+1][u.dims[1]], 1, MPI_COL, top.nbrup, 0, 
                 &u.arr[odd+1][0], 1, MPI_COL, top.nbrdwn, 0, top.comm, MPI_STATUS_IGNORE);
    
    MPI_Sendrecv(&u.arr[odd+1][1], 1, MPI_COL, top.nbrdwn, 0, 
                 &u.arr[odd+1][u.dims[1]+1], 1, MPI_COL, top.nbrup, 0, top.comm, MPI_STATUS_IGNORE);
}

// preconditioned conjugate gradient algorithm //
void precond_CG(Grid u, MPI_Topology top){
    Grid_ptr rnew, rold, znew, zold, tmp_ptr; 
    Grid tmp, p, Ap;
    double alpha, beta;
    double loc_sum, glob_sum;
    double tmpsum1=0, tmpsum2=0;
    double gtmpsum1=0, gtmpsum2=0;
    int count=0;
    int nx = u.dims[0], ny = u.dims[1];
    
    rnew = (Grid_ptr)malloc(sizeof(Grid));
    rold = (Grid_ptr)malloc(sizeof(Grid));
    znew = (Grid_ptr)malloc(sizeof(Grid));
    zold = (Grid_ptr)malloc(sizeof(Grid));

    *rnew   = init_grid(nx, ny);
    *rold   = init_grid(nx, ny);
    *znew   = init_grid(nx, ny);
    *zold   = init_grid(nx, ny);
    tmp     = init_grid(nx, ny);
    p       = init_grid(nx, ny);
    Ap      = init_grid(nx, ny);   
    
    residual(*rold, u, top);        // b-Ax
    rb_GS(*zold, *rold, top);       // z = M^(-1)r (applying preconditioning)
    memcpy(p.vals, zold->vals, (nx+2)*(ny+2)*sizeof(double));   // p=z
    
    // while until convergence //
    while(count++ < MAXITERS){
        laplace(Ap, p, top);    // getting A*p
        
        tmpsum1 = frob_inner_prod(*rold, *zold);    // <r,z>
        tmpsum2 = frob_inner_prod(Ap, p);           // <A*p,p>
        MPI_Allreduce(&tmpsum1, &gtmpsum1, 1, MPI_DOUBLE, MPI_SUM, top.comm);
        MPI_Allreduce(&tmpsum2, &gtmpsum2, 1, MPI_DOUBLE, MPI_SUM, top.comm);
        alpha = gtmpsum1/gtmpsum2;      // alpha = <r,z>/<A*p,p>

        vec_scal_prod(tmp, p, alpha);   // alpha*p
        vec_vec_add(u, u, tmp);         // x += alpha*p
        
        vec_scal_prod(tmp, Ap, -alpha); // -alpha*A*p
        vec_vec_add(*rnew, *rold, tmp); // r -= alpha*A*p
        
        loc_sum = grid_diff(*rnew);     // checking global convergence
        MPI_Allreduce(&loc_sum, &glob_sum, 1, MPI_DOUBLE, MPI_SUM, top.comm); 
        if(glob_sum<ERR)
            break;
        
        rb_GS(*znew, *rnew, top);       // z = M^(-1)r
        
        tmpsum1 = frob_inner_prod(*rnew, *znew);
        tmpsum2 = frob_inner_prod(*rold, *zold);
        MPI_Allreduce(&tmpsum1, &gtmpsum1, 1, MPI_DOUBLE, MPI_SUM, top.comm);
        MPI_Allreduce(&tmpsum2, &gtmpsum2, 1, MPI_DOUBLE, MPI_SUM, top.comm);
        beta = gtmpsum1/gtmpsum2;       // beta = <znew,rnew>/<rold,zold>
        
        vec_scal_prod(tmp, p, beta);    // p = z + beta*p
        vec_vec_add(p, *znew, tmp);

        tmp_ptr = rnew;
        rnew = rold;
        rold = tmp_ptr;

        tmp_ptr = znew;
        znew = zold;
        zold = tmp_ptr;
    }
    
    if(top.coords[0]==0 && top.coords[1]==0 && count == MAXITERS+1)
        printf("Max iterations reached in C-G.\n");
    
    free_grid(rnew); free_grid(rold); free_grid(znew); free_grid(zold);
    free(rold); free(rnew); free(zold); free(znew);
    free_grid(&Ap); free_grid(&p); free_grid(&tmp);
}
