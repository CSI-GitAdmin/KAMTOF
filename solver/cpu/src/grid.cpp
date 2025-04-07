#include "grid.h"

#include <cstdio>
#include <vector>
#include <array>
#include <cmath>
#include <sys/time.h>

Grid::Grid():
   xcen(m_silo.register_entry<strict_fp_t, CDF::StorageType::VECTOR>("grid_xcen")),
   volume(m_silo.register_entry<strict_fp_t, CDF::StorageType::CELL>("grid_volume")),
   number_of_neighbors(m_silo.register_entry<int, CDF::StorageType::VECTOR>("grid_number_of_neighbors")),
   cell_neighbors(m_silo.register_entry<int, CDF::StorageType::FACE>("grid_cell_neighbors")),
   area(m_silo.register_entry<strict_fp_t, CDF::StorageType::FACE>("grid_area")),
   normal(m_silo.register_entry<strict_fp_t, CDF::StorageType::VECTOR>("grid_normal")),
   boundary_face_to_cell(m_silo.register_entry<int, CDF::StorageType::BOUNDARY>("grid_boundary_face_to_cell")),
   boundary_xcen(m_silo.register_entry<strict_fp_t, CDF::StorageType::VECTOR>("grid_boundary_xcen")),
   boundary_area(m_silo.register_entry<strict_fp_t, CDF::StorageType::BOUNDARY>("grid_boundary_area")),
   boundary_normal(m_silo.register_entry<strict_fp_t, CDF::StorageType::VECTOR>("grid_boundary_normal")),
   boundary_type_start_and_end_index(m_silo.register_entry<int, CDF::StorageType::VECTOR>("grid_boundary_type_start_and_end_index"))
{}

void Grid::generate_grid(const int Nx, const int Ny)
{
   this->num_cells = Nx*Ny;
   this->num_faces = 4*(Nx-2)*(Ny-2) + 3*(2*(Nx-2)+2*(Ny-2)) + 2*4;
   this->num_boundary_faces = 2*Nx + 2*Ny;
   this->num_boundary = 4;

   dx = 1.0 / Nx;
   dy = 1.0 / Ny;
   
   //calculating (x, y) location for each cell row by row starting from bottom left corner cell
   xcen.resize(this->num_cells * 2);
   for (int i = 0; i < Nx; i++)
   {
      for (int j = 0; j < Ny; j++)
      {
         const int index = (j * Nx) + i;
         xcen[2 * index] = dx * 0.5 + (i * dx);
         xcen[(2 * index) + 1] = (dy * 0.5) + (j * dy);
      }
   }
   
   //calculating volume of each cell
   for (unsigned int i = 0; i < this->num_cells; i++)
   {
      volume[i] = dx * dy * 1.0;
   }
   
   // Generate neighbor connectivity

   number_of_neighbors.resize(this->num_cells + 1);
   number_of_neighbors[0] = 0;
   
   for (int i = 1; i < Nx - 1; i++) // Interior cells
   {
      for (int j = 1; j < Ny - 1; j++)
      {
         const int index = 2*2 + 3*(Nx-2) + 3*(1+2*(j-1)) + 4*(j-1)*(Nx-2) + 4*(i-1);
         const int cell = (j * Nx) + i;
         cell_neighbors[index + 0] = cell - 1;  // Left neighbor
         cell_neighbors[index + 1] = cell + 1;  // Right neighbor
         cell_neighbors[index + 2] = cell - Nx; // Bottom neighbor
         cell_neighbors[index + 3] = cell + Nx; // Top neighbor
         number_of_neighbors[cell+1] = index + 4;
      }
   }
   
   for (int i = 1; i < Nx - 1; i++)
   {
      {  // Bottom row except corners
         const int index = 2*1 + 3*(i-1);
         const int j = 0;
         const int cell = (j * Nx) + i;
         cell_neighbors[index + 0] = cell - 1;
         cell_neighbors[index + 1] = cell + 1;
         cell_neighbors[index + 2] = cell + Nx;
         number_of_neighbors[cell+1] = index + 3;
      }
      
      {  // Top row except corners
         const int index = 2*3 + 3*(Nx-2) + 4*(Nx-2)*(Ny-2) + 2*3*(Ny-2) + 3*(i-1);
         const int j = Ny - 1;
         const int cell = (j * Nx) + i;
         cell_neighbors[index + 0] = cell - 1;
         cell_neighbors[index + 1] = cell + 1;
         cell_neighbors[index + 2] = cell - Nx;
         number_of_neighbors[cell+1] = index + 3;
      }
   }
   
   for (int j = 1; j < Ny - 1; j++)
   {
      {  // Left column except corners
         const int index = 2*2 + 3*(Nx-2) + 3*2*(j-1) + 4*(j-1)*(Nx-2);
         const int i = 0;
         const int cell = (j * Nx) + i;
         cell_neighbors[index + 0] = cell + 1;
         cell_neighbors[index + 1] = cell - Nx;
         cell_neighbors[index + 2] = cell + Nx;
         number_of_neighbors[cell+1] = index + 3;
      }
      
      {  // Right column except corners
         const int index = 2*2 + 3*(Nx-2) + 3*(2*(j-1)+1) + 4*(Nx-2)*j;
         const int i = Nx - 1;
         const int cell = (j * Nx) + i;
         cell_neighbors[index + 0] = cell - 1;
         cell_neighbors[index + 1] = cell - Nx;
         cell_neighbors[index + 2] = cell + Nx;
         number_of_neighbors[cell+1] = index + 3;
      }
   }
   
   // Four corners
   {  // Bottom left corner
      const int index = 0;
      const int i = 0;
      const int j = 0;
      const int cell = (j * Nx) + i;
      cell_neighbors[index + 0] = cell + 1;
      cell_neighbors[index + 1] = cell + Nx;
      number_of_neighbors[cell+1] = index + 2;
   }
   
   {  // Bottom right corner
      const int index = 2 + 3*(Nx-2);
      const int i = Nx - 1;
      const int j = 0;
      const int cell = (j * Nx) + i;
      cell_neighbors[index + 0] = cell - 1;
      cell_neighbors[index + 1] = cell + Nx;
      number_of_neighbors[cell+1] = index + 2;
   }
   
   {  // Top left corner
      const int index = 2*2 + 3*(Nx-2) + 4*(Nx-2)*(Ny-2) + 3*2*(Ny-2);
      const int i = 0;
      const int j = Ny - 1;
      const int cell = (j * Nx) + i;
      cell_neighbors[index + 0] = cell + 1;
      cell_neighbors[index + 1] = cell - Nx;
      number_of_neighbors[cell+1] = index + 2;
   }
   
   {  // Top right corner
      const int index = 2*3 + 4*(Nx-2)*(Ny-2) + 3*2*(Ny-2) + 3*2*(Nx-2);
      const int i = Nx - 1;
      const int j = Ny - 1;
      const int cell = (j * Nx) + i;
      cell_neighbors[index + 0] = cell - 1;
      cell_neighbors[index + 1] = cell - Nx;
      number_of_neighbors[cell+1] = index + 2;
   }

   normal.resize(2 * this->num_faces);
   for (int i = 0; i < this->num_cells; i++)
   {
      for (int j = number_of_neighbors[i]; j < number_of_neighbors[i + 1]; j++)
      {
         const int left_cell = i;
         const int right_cell = cell_neighbors[j];
         
         const strict_fp_t delta_x = xcen[(2 * right_cell)] - xcen[(2 * left_cell)];
         const strict_fp_t delta_y = xcen[(2 * right_cell) + 1] - xcen[(2 * left_cell) + 1];

         const strict_fp_t length = sqrt((delta_x * delta_x) + (delta_y * delta_y));
         normal[2*j] = delta_x / length;
         normal[2*j+1] = delta_y / length;

         if ((std::abs(normal[2*j]) > 0.5) && (std::abs(normal[2*j+1]) < 0.5))
            area[j] = dy * 1.0;
         else
            area[j] = dx * 1.0;
      }
   }
   
   // Generate boundary information
   int counter = 0;
   
   const int number_of_boundary_faces = Nx + Nx + Ny + Ny;
   
   boundary_xcen.resize (this->num_boundary_faces * 2);
   boundary_normal.resize (this->num_boundary_faces * 2);
   boundary_type_start_and_end_index.resize(this->num_boundary + 1);
   
   boundary_type_start_and_end_index[0] = counter;
   for (int i = 0; i < Nx; i++)
   {
      int j = 0;
      const int cell = (j * Nx) + i;
      boundary_face_to_cell[counter] = cell;
      boundary_xcen[(2 * counter)] = xcen[(2 * cell)];
      boundary_xcen[(2 * counter) + 1] = 0.0;
      boundary_normal[2 * counter] = 0.0;
      boundary_normal[(2 * counter) + 1] = -1.0;
      boundary_area[counter] = dx * 1.0;
      counter++;
   }
   boundary_type_start_and_end_index[1] = counter;
   
   for (int i = 0; i < Nx; i++)
   {
      int j = Ny - 1;
      const int cell = (j * Nx) + i;
      boundary_face_to_cell[counter] = cell;
      boundary_xcen[(2 * counter)] = xcen[(2 * cell)];
      boundary_xcen[(2 * counter) + 1] = 1.0;
      boundary_normal[2 * counter] = 0.0;
      boundary_normal[(2 * counter) + 1] = 1.0;
      boundary_area[counter] = dx * 1.0;
      counter++;
   }
   boundary_type_start_and_end_index[2] = counter;
   
   for (int j = 0; j < Ny; j++)
   {
      int i = 0;
      const int cell = (j * Nx) + i;
      boundary_face_to_cell[counter] = cell;
      boundary_xcen[(2 * counter)] = 0.0;
      boundary_xcen[(2 * counter) + 1] = xcen[(2 * cell) + 1];
      boundary_normal[2 * counter] = -1.0;
      boundary_normal[(2 * counter) + 1] = 0.0;
      boundary_area[counter] = dy * 1.0;
      counter++;
   }
   boundary_type_start_and_end_index[3] = counter;
   
   for (int j = 0; j < Ny; j++)
   {
      int i = Nx - 1;
      const int cell = (j * Nx) + i;
      boundary_face_to_cell[counter] = cell;
      boundary_xcen[(2 * counter)] = 1.0;
      boundary_xcen[(2 * counter) + 1] = xcen[(2 * cell) + 1];
      boundary_normal[2 * counter] = 1.0;
      boundary_normal[(2 * counter) + 1] = 0.0;
      boundary_area[counter] = dy * 1.0;
      counter++;
   }
   boundary_type_start_and_end_index[4] = counter;
}


void Grid::print_grid_information ()
{
   for (unsigned int i = 0; i < this->num_cells; i++)
   {
      printf("Cell %d: x %f y %f\n", i, xcen[2*i], xcen[(2*i) + 1]);
   }
   
   for (unsigned int i = 0; i < this->num_cells; i++)
   {
      printf("Cell %d: Neighbors ", i);
      for (unsigned int j = number_of_neighbors[i]; j < number_of_neighbors[i+1]; j++)
      {
         printf(" %d ", cell_neighbors[j]);
      }
      printf("\n");
   }
   
   for (unsigned int i = 0; i < this->num_boundary; i++)
   {
      printf("Boundary %d: Start %d and end %d\n", i, boundary_type_start_and_end_index[i], 
             boundary_type_start_and_end_index[i+1]);
   }
   
   for (unsigned int i = 0; i < boundary_face_to_cell.size(); i++)
   {
      printf("Boundary face %d: Cell %d\n", i, boundary_face_to_cell[i]);
   }
   
   for (unsigned int i = 0; i < boundary_face_to_cell.size(); i++)
   {
      printf("Boundary face %d: x %f y %f\n", i, boundary_xcen[2*i], boundary_xcen[(2*i) + 1]);
   }
   
   for (unsigned int i = 0; i < boundary_area.size(); i++)
   {
      printf("Boundary face %d: Normal %f %f Area %f\n", i, boundary_normal[2*i], 
             boundary_normal[2*i+1], boundary_area[i]);
   }
}
