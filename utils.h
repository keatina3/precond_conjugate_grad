#ifndef _UTILS_H_
#define _UTILS_H_

typedef struct {
    int nprocs;
    int nbrup, nbrdwn, nbrleft, nbrright;
    int myid;
} Topology;

typedef struct {
    double  *vals;
    double  **arr;
    int     n;
    int     m;
} Grid, *Grid_ptr;

Topology init_top(int nprocs, int myid, MPI_Cart comm);

Grid init_grid(int n, int m);
void free_grid(Grid_ptr x);
double frob_inner_prod(Grid x, Grid y);
void vec_scal_prod(Grid ax, Grid x, double a);
void vec_vec_add(Grid xy, Grid x, Grid y);

#endif
