#ifndef GRID_H
#define GRID_H

#include <vector>
#include <array>

#include "silo.h"
#include "silo_fwd.h"

class Grid
{
public:
   void generate_grid (const int Nx, const int Ny);
   
   void print_grid_information ();
   
   int get_number_of_cells() const
   {
      return this->num_cells;
   }
   
   int get_number_of_boundary_faces() const
   {
      return boundary_face_to_cell.size();
   }

   int get_number_of_faces() const
   {
      return area.size();
   }

   int get_number_of_boundaries() const
   {
      return this->num_boundary;
   }
   
   VectorRead<int>& get_boundary_type_start_and_end_index () const
   {
      return boundary_type_start_and_end_index;
   }
   
   Boundary<int>& get_boundary_face_to_cell () const
   {
      return boundary_face_to_cell;
   }
   
   CellRead<strict_fp_t>& get_volume () const
   {
      return volume;
   }
   
   VectorRead<strict_fp_t>& get_xcen () const
   {
      return xcen;
   }
   
   VectorRead<strict_fp_t>& get_boundary_xcen () const
   {
      return boundary_xcen;
   }
   
   VectorRead<strict_fp_t>& get_boundary_normal () const
   {
      return boundary_normal;
   }
   
   BoundaryRead<strict_fp_t>& get_boundary_area () const
   {
      return boundary_area;
   }
   
   FaceRead<int>& get_cell_neighbors () const
   {
      return cell_neighbors;
   }
   
   VectorRead<int>& get_number_of_neighbors () const
   {
      return number_of_neighbors;
   }
   
   FaceRead<strict_fp_t>& get_area () const
   {
      return area;
   }
   
   VectorRead<strict_fp_t>& get_normal () const
   {
      return normal;
   }

   Grid();
   
private:
   int num_cells;
   int num_faces;
   int num_boundary_faces;
   int num_boundary;
   strict_fp_t dx, dy;
   Vector<strict_fp_t> xcen; // 2*num_cells
   Cell<strict_fp_t> volume; // num_cells
   Vector<int> number_of_neighbors; // num_cells+1
   Face<int> cell_neighbors; // num_faces
   Face<strict_fp_t> area; // num_faces
   Vector<strict_fp_t> normal; // 2*num_faces
   Boundary<int> boundary_face_to_cell; // num_boundary_faces
   Vector<int> boundary_type_start_and_end_index; // 2*num_boundary_faces
   Vector<strict_fp_t> boundary_xcen; // 2*num_boundary_faces
   Vector<strict_fp_t> boundary_normal; // 2*num_boundary_faces
   Boundary<strict_fp_t> boundary_area; // num_boundary_faces
};

#endif /* GRID_H */

