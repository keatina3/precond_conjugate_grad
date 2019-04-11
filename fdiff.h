#ifndef _FDIFF_H_
#define _FDIFF_H_

#define MAXITERS 1000
#define ERR 1.0E-07

void init_boundaries(Grid u, MPI_Topology top);
void laplace(Grid Lx, Grid x, MPI_Topology top);
void residual(Grid r, Grid u, MPI_Topology top);
void GS_solve(Grid x, Grid b, MPI_Topology top, int i, int j);
void laplace_rb_GS(Grid x, Grid b, MPI_Topology);
void exchange(Grid u, int odd, MPI_Topology comm, MPI_Datatype MPI_ROW, MPI_Datatype MPI_COL);
void precond_CG(Grid u, MPI_Topology top);

#endif
