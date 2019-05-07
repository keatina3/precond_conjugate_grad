#ifndef _UTILS_H_
#define _UTILS_H_

typedef struct {
    MPI_Comm comm;
    int nprocs;
    int coords[2];
    int nbrup, nbrdwn, nbrleft, nbrright;
} MPI_Topology;

typedef struct {
    double  *vals;
    double  **arr;
    int     dims[2];
} Grid, *Grid_ptr;

MPI_Topology init_top(int nprocs, int myid, MPI_Comm comm);
int decomp1d(int n, int size, int rank, int *s, int *e);

Grid init_grid(int n, int m);
void free_grid(Grid_ptr x);
double grid_diff(Grid x);
double frob_inner_prod(Grid x, Grid y);
void vec_scal_prod(Grid ax, Grid x, double a);
void vec_vec_add(Grid x_y, Grid x, Grid y);
void print_grid(Grid x, MPI_Topology top);
void print_col(Grid x, int col, int yproc, int xprocs, MPI_Topology top);

#endif
