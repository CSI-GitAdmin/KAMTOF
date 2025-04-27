#ifndef GRID_H
#define GRID_H

#include <vector>
#include <array>

#include "silo.h"
#include "silo_fwd.h"

class Grid
{
public:
   void allocate_variables();

   void generate_grid();
   
   void distribute_grid();

   void assign_mpi_grid();

   void print_grid_information ();
   
   Grid(const int Nx, const int Ny);

   int Nx;
   int Ny;

   std::vector<int>num_solved_array;
   std::vector<int>num_involved_array;

   int num_cells;
   int num_faces;
   int num_solved;
   int num_involved;
   int num_attached;
   
   int num_boundary_faces;
   int num_boundary;
   strict_fp_t dx, dy;
};

void mpi_nbnb_transfer(strict_fp_t *vec);

#endif /* GRID_H */

