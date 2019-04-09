#include <stdlib.h>
#include <string.h>
#include "utils.h"
#include "fdiff.h"

void laplace(Grid Lx, Grid x){
    int i,j;
    
    for(i=0;i<(x.n+1);i++){
        for(j=0;j<(x.m+1);j++){
            Lx[i][j] = -4*x[i][j] 
            if(i>0)
                Lx[i][j] += x[i-1][j] 
            if(i<x.n)
                Lx[i][j] += x[i+1][j] 
            if(j>o)
                Lx[i][j] += x[i][j-1] 
            if(j<x.m)
                Lx[i][j] += x[i][j+1];
        }
    }
}

// need to incorporate uold //
void residual(Grid r, u){
    int i,j;
    for(i=1;i<u.n;i++)
        for(j=1;j<u.m;j++)
            r[i][j] = 4*u[i][j] - u[i+1][j] - u[i][j+1] - r[i][j-1] - r[i-1][j]; 

}

void laplace_rb_GS(){
    int i,j;
    
    while(count<MAXITERS){
        for(i=1;i<u.n;i++){
            for(i=j;j<u.m;j++){
                if((i+j)%2==0){
                    u[i][j] = u[i][j] + 0.25*(-4*u[i][j] + u[i-1][j] + u[i+1][j] 
                                    + u[i][j-1] + u[i][j+1];
                }
            }
        }

        exchange();
        MPI_Barrier();
        for(i=1;i<u.n;i++){
            for(i=j;j<u.m;j++){
                if((i+j)%2==1){
                    u[i][j] = u[i][j] + 0.25*(-4*u[i][j] + u[i-1][j] + u[i+1][j] 
                                    + u[i][j-1] + u[i][j+1];
                }
            }
        }
        exchange();
        MPI_Barrier();
        
        if(GLOB CONVERGE)
            break;
    }
}

void precond_CG(Grid u){
    Grid rnew, rold, znew, zold, p, Ap;
    double alpha, beta;
    int nx = u.nx, ny = u.ny;

    rnew    = init_grid(nx, ny);
    rold    = init_grid(nx, ny);
    znew    = init_grid(nx, ny);
    rold    = init_grid(nx, ny);
    p       = init_grid(nx, ny);
    Ap      = init_grid(nx, ny);    
    
    residual(rold, u);
    laplace_rb_GS(z, r)
    memcpy(p.vals, z.vals, (nx+2)*(ny+2)*sizeof(double));
    while(count < MAXITERS){
        laplace(Ap, p);         // need to fix for boundary vals //
        alpha = frob_inner_prod(rold, zold) / frob_inner_prod(Ap, p);
        x = vec_vec_add(x, vec_scal_prod(p_vals, alpha));
        rnew = vec_vec_add(rold, vec_scal_prod(Ap, -alpha));
        if(grid_diff(rnew) < ERR)
            break;
        laplace_rb_gs(znew, rnew);
        beta = frob_inner_prod(rnew, znew) / frob_inner_prod(rold, zold);
        p = vec_vec_add(znew, vec_scal_prod(p, beta));
    }
}
