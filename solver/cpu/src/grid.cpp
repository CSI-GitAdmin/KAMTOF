#include <cstdio>
#include <vector>
#include <array>
#include <cmath>
#include <sys/time.h>
#include <set>

#include "cpu_globals.h"
#include "grid.h"
#include "mpiclass.h"

#ifdef ENABLE_GPU
#include "gpu_api_functions.h"
#endif

Grid::Grid(const int Nx, const int Ny)
{
   this->Nx = Nx;
   this->Ny = Ny;

   num_solved = 0;
   num_involved = 0;
   num_faces = 0;
   num_attached = 0;
}

void Grid::allocate_variables()
{
   m_silo.register_entry<int, CDF::StorageType::VECTOR>("number_of_neighbors");
   m_silo.register_entry<int, CDF::StorageType::VECTOR>("ranks_list");
   m_silo.register_entry<int, CDF::StorageType::VECTOR>("cell_neighbors");
   
   m_silo.register_entry<int, CDF::StorageType::VECTOR>("boundary_type_start_and_end_index");

   m_silo.register_entry<strict_fp_t, CDF::StorageType::CELL>("volume_local");
   m_silo.register_entry<strict_fp_t, CDF::StorageType::VECTOR>("xcen_local");
   m_silo.register_entry<strict_fp_t, CDF::StorageType::FACE>("area_local");

   m_silo.register_entry<int, CDF::StorageType::VECTOR>("number_of_neighbors_local");
   m_silo.register_entry<int, CDF::StorageType::FACE>("cell_neighbors_local");
   m_silo.register_entry<strict_fp_t, CDF::StorageType::VECTOR>("normal_local");

   m_silo.register_entry<int, CDF::StorageType::BOUNDARY>("boundary_face_to_cell_local");
   m_silo.register_entry<strict_fp_t, CDF::StorageType::BOUNDARY>("boundary_area_local");
   m_silo.register_entry<strict_fp_t, CDF::StorageType::VECTOR>("boundary_xcen_local");
   m_silo.register_entry<strict_fp_t, CDF::StorageType::VECTOR>("boundary_normal_local");

   m_silo.register_entry<int, CDF::StorageType::VECTOR>("local_global");
   m_silo.register_entry<int, CDF::StorageType::VECTOR>("global_local");
}

void Grid::generate_grid()
{
   this->num_cells = Nx*Ny;
   this->num_faces = 4*(Nx-2)*(Ny-2) + 3*(2*(Nx-2)+2*(Ny-2)) + 2*4;
   this->num_boundary_faces = 2*Nx + 2*Ny;
   this->num_boundary = 4;

   dx = 1.0 / Nx;
   dy = 1.0 / Ny;
   
   // Generate neighbor connectivity
   Vector<int> number_of_neighbors = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("number_of_neighbors");
   Vector<int> cell_neighbors = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("cell_neighbors");
   cell_neighbors.resize(4*(Nx-2)*(Ny-2) + 3*(2*(Nx-2)+2*(Ny-2)) + 2*4);
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

   // MPI grid distribution, ranks_list will give the rank each grid point is associated with
   distribute_grid();

   assign_mpi_grid();

   m_silo.resize<CDF::StorageType::CELL>(num_involved);
   m_silo.resize<CDF::StorageType::FACE>(num_faces);
   m_silo.resize<CDF::StorageType::BOUNDARY>(num_attached);

   // Set up local neighbors
   Vector<int> number_of_neighbors_local = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("number_of_neighbors_local");
   VectorRead<int> local_global = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("local_global");
   number_of_neighbors_local.resize(num_solved + 1);
   number_of_neighbors_local[0] = 0;
   for (int i = 0; i < num_solved; i++)
   {
      const int ig = local_global[i];
      number_of_neighbors_local[i + 1] = number_of_neighbors_local[i] + number_of_neighbors[ig + 1] - number_of_neighbors[ig];
   }

   VectorRead<int> global_local = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("global_local");
   Face<int> cell_neighbors_local = m_silo.retrieve_entry<int, CDF::StorageType::FACE>("cell_neighbors_local");
   for (int i = 0; i < num_solved; i++)
   {
      const int start_index = number_of_neighbors_local[i];
      const int ig = local_global[i];
      for (int j = number_of_neighbors[ig]; j < number_of_neighbors[ig + 1]; j++)
      {
         cell_neighbors_local[start_index + j - number_of_neighbors[ig]] = global_local[cell_neighbors[j]];
      }
   }

   // Calculate (x, y) location for each cell
   VectorRead<int> ranks_list = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("ranks_list");
   Vector<strict_fp_t> xcen_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::VECTOR>("xcen_local");
   std::vector<strict_fp_t> loc_x(num_involved), loc_y(num_involved);
   xcen_local.resize(2 * num_involved);
   for (int i = 0; i < Nx; i++)
   {
      for (int j = 0; j < Ny; j++)
      {
         const int cell = (j * Nx) + i;
         if(ranks_list[cell] == rank)
         {
            const int cell_local = global_local[cell];
            loc_x[cell_local] = dx * 0.5 + (i * dx);
            loc_y[cell_local] = (dy * 0.5) + (j * dy);
         }
      }
   }

   mpi_nbnb_transfer(loc_x.data());
   mpi_nbnb_transfer(loc_y.data());

   for(int i = 0; i < num_involved; i++)
   {
      xcen_local[2 * i] = loc_x[i];
      xcen_local[2 * i + 1] = loc_y[i];
   }

   // Set up local boundary variables
   Vector<int> boundary_type_start_and_end_index = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("boundary_type_start_and_end_index");
   Boundary<int> boundary_face_to_cell_local = m_silo.retrieve_entry<int, CDF::StorageType::BOUNDARY>("boundary_face_to_cell_local");
   Boundary<strict_fp_t> boundary_area_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::BOUNDARY>("boundary_area_local");
   Vector<strict_fp_t> boundary_xcen_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::VECTOR>("boundary_xcen_local");
   Vector<strict_fp_t> boundary_normal_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::VECTOR>("boundary_normal_local");
   boundary_type_start_and_end_index.resize(this->num_boundary + 1);
   boundary_normal_local.resize(2 * num_attached);
   boundary_xcen_local.resize(2 * num_attached);
   
   int counter = 0;
   boundary_type_start_and_end_index[0] = counter;
   for(int i = 0; i < Nx; i++) // bottom boundary
   {
      const int j = 0;
      const int cell = (j * Nx) + i;
      if(ranks_list[cell] == rank)
      {
         const int cell_local = global_local[cell];
         boundary_face_to_cell_local[counter] = cell_local;
         boundary_area_local[counter] = dx;
         boundary_xcen_local[(2 * counter)] = xcen_local[(2 * cell_local)];
         boundary_xcen_local[(2 * counter) + 1] = 0.0;
         boundary_normal_local[2 * counter] = 0.0;
         boundary_normal_local[(2 * counter) + 1] = -1.0;
         counter++;
      }
   }
   boundary_type_start_and_end_index[1] = counter;

   for(int i = 0; i < Nx; i++) // top boundary
   {
      const int j = Ny - 1;
      const int cell = (j * Nx) + i;
      if(ranks_list[cell] == rank)
      {
         const int cell_local = global_local[cell];
         boundary_face_to_cell_local[counter] = cell_local;
         boundary_area_local[counter] = dx;
         boundary_xcen_local[(2 * counter)] = xcen_local[(2 * cell_local)];
         boundary_xcen_local[(2 * counter) + 1] = 1.0;
         boundary_normal_local[2 * counter] = 0.0;
         boundary_normal_local[(2 * counter) + 1] = 1.0;
         counter++;
      }
   }
   boundary_type_start_and_end_index[2] = counter;

   for(int j = 0; j < Ny; j++) // left boundary
   {
      const int i = 0;
      const int cell = (j * Nx) + i;
      if(ranks_list[cell] == rank)
      {
         const int cell_local = global_local[cell];
         boundary_face_to_cell_local[counter] = cell_local;
         boundary_area_local[counter] = dy;
         boundary_xcen_local[(2 * counter)] = 0.0;
         boundary_xcen_local[(2 * counter) + 1] = xcen_local[(2 * cell_local) + 1];
         boundary_normal_local[2 * counter] = -1.0;
         boundary_normal_local[(2 * counter) + 1] = 0.0;
         counter++;
      }
   }
   boundary_type_start_and_end_index[3] = counter;

   for(int j = 0; j < Ny; j++) // right boundary
   {
      const int i = Nx - 1;
      const int cell = (j * Nx) + i;
      if(ranks_list[cell] == rank)
      {
         const int cell_local = global_local[cell];
         boundary_face_to_cell_local[counter] = cell_local;
         boundary_area_local[counter] = dy;
         boundary_xcen_local[(2 * counter)] = 1.0;
         boundary_xcen_local[(2 * counter) + 1] = xcen_local[(2 * cell_local) + 1];
         boundary_normal_local[2 * counter] = 1.0;
         boundary_normal_local[(2 * counter) + 1] = 0.0;
         counter++;
      }
   }
   boundary_type_start_and_end_index[4] = counter;

   // Calculate local cell volume
   Cell<strict_fp_t> volume_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("volume_local");
   for (unsigned int il = 0; il < num_solved; il++)
   {
      volume_local[il] = dx * dy * 1.0;
   }

   // Calculate local face area and normal
   Face<strict_fp_t> area_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::FACE>("area_local");
   Vector<strict_fp_t> normal_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::VECTOR>("normal_local");
   normal_local.resize(2 * num_faces);
   for (int il = 0; il < num_solved; il++)
   {
      for (int jl = number_of_neighbors_local[il]; jl < number_of_neighbors_local[il + 1]; jl++)
      {
         const int left_cell = il;
         const int right_cell = cell_neighbors_local[jl];
         
         const strict_fp_t delta_x = xcen_local[(2 * right_cell)] - xcen_local[(2 * left_cell)];
         const strict_fp_t delta_y = xcen_local[(2 * right_cell) + 1] - xcen_local[(2 * left_cell) + 1];

         const strict_fp_t length = sqrt((delta_x * delta_x) + (delta_y * delta_y));
         normal_local[2*jl] = delta_x / length;
         normal_local[2*jl+1] = delta_y / length;

         if ((std::abs(normal_local[2*jl]) > 0.5) && (std::abs(normal_local[2*jl+1]) < 0.5))
            area_local[jl] = dy * 1.0;
         else
            area_local[jl] = dx * 1.0;
      }
   }
}

void Grid::distribute_grid()
{
   const int number_of_cells = Nx * Ny;

   Vector<int> ranks_list = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("ranks_list");
   ranks_list.resize (number_of_cells); 
   int size = numprocs;
   // split x-direction amongst procesors
   int base_size = Nx / size;
   int remainder = Nx % size;
   std::vector<int>assign_1d_rank(Nx);
   std::vector<int>size_rank(size);
   //size rank will give the number of points distributed for each rank in x-direction
   for(int ii = 0; ii < size; ii++)
   {
      size_rank[ii] = base_size;	   
   }
   for(int ii = 0; ii < remainder; ii++)
   {
      size_rank[ii] = base_size + 1;	   
   }
   // now distribute each point on the x-direction amongst the processors
   int count = 0;
   for(int ii = 0; ii < size; ii++)
   {
      for (int jj = 0; jj < size_rank[ii]; jj++)
      {
         assign_1d_rank[count++] = ii;	      
      }
   }
   // using the 1D distribution, assign each point on the 2D grid to a processor
   for (int i = 0; i < Nx; i++)
   {
      for (int j = 0; j < Ny; j++)
      {
         const int index = (j * Nx) + i;
	      ranks_list[index] = assign_1d_rank[i];
      }
   }
}

void Grid::assign_mpi_grid()
{
   const int neq = this->num_cells;
   VectorRead<int> ranks_list = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("ranks_list");
   Vector<int> local_global = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("local_global");
   Vector<int> global_local = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("global_local");

   num_solved_array.resize(numprocs);
   num_involved_array.resize(numprocs);
   local_global.resize(neq);
   global_local.resize(neq);

   memset(num_solved_array.data(), 0, numprocs*sizeof(int));
   memset(num_involved_array.data(), 0, numprocs*sizeof(int));

   int *offsets_array = new int[numprocs + 1];
   int *involved_array = new int[numprocs];
   int *gcnum = new int[neq];
   int *gcinvnum = new int[neq];
 
   int ssize;
   int rsize;
   strict_fp_t *sbuf;
   strict_fp_t *rbuf;
   int *slist;
   int *rlist;
   int *sdisp;
   int *rdisp;
   int *scounts;
   int *rcounts;
   
   rdisp = new int[numprocs + 1]();
   sdisp = new int[numprocs + 1]();
   rcounts = new int[numprocs]();
   scounts = new int[numprocs]();
   ssize = 0;
   rsize = 0;

   // arbitrary initialization
   for (int nn = 0; nn < neq; nn++)
   {
      global_local[nn] = -1000;
      local_global[nn] = -1000;
   }

   // get the number of nodes on this rank
   for (int ii = 0; ii < neq; ii++)
   {
      num_solved_array[ranks_list[ii]]++;
      if (ranks_list[ii] == rank)
      {
         local_global[num_solved] = ii;
         global_local[ii] = num_solved;
         num_solved++;
      }
   }


   MPI_Allgather(&num_solved, 1, MPI_INT, num_solved_array.data(), 1, MPI_INT, MPI_COMM_WORLD);

   offsets_array[0] = 0;
   for (int ii = 0; ii < numprocs; ii++)
   {
      involved_array[ii] = 0;
      offsets_array[ii + 1] = offsets_array[ii] + num_solved_array[ii];
   }

   // get the offset for the current rank
   int offset = 0;
   for (int ii = 0; ii < rank; ii++)
   {
      offset = offset + num_solved_array[ii];
   }

   // get different global node numbering
   // gcnum gives the new global number
   // gcinvnum maps the new global number to the old global number
   for (int ii = 0; ii < neq; ii++)
   {
      int lr = ranks_list[ii];
      int jj = offsets_array[lr] + involved_array[lr];
      gcnum[ii] = jj;
      gcinvnum[jj] = ii;
      involved_array[lr]++;
   }

   // get the number of processor boundary nodes
   // get num_involved
   VectorRead<int> number_of_neighbors = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("number_of_neighbors");
   VectorRead<int> cell_neighbors = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("cell_neighbors");
   std::set<int> involved_nodes_set; // To track unique gcnum values
   for (int nn = 0; nn < neq; nn++)
   {
      if (ranks_list[nn] != rank)
         continue;
      
      for(int j = number_of_neighbors[nn]; j < number_of_neighbors[nn+1]; j++)
      {
         const int i = cell_neighbors[j];
         if(ranks_list[i] != rank)
         {
            int ng = gcnum[i];
            if(involved_nodes_set.insert(ng).second)
            {
               rcounts[ranks_list[i]]++;
               num_involved++;
            }
         }
      }
   }
   int involved_node = involved_nodes_set.size();

   rlist = new int[involved_node]();
   slist = new int[involved_node]();

   std::copy(involved_nodes_set.begin(), involved_nodes_set.end(), rlist);

   num_involved = num_solved + involved_node;
   MPI_Allgather(&num_involved, 1, MPI_INT, num_involved_array.data(), 1, MPI_INT, MPI_COMM_WORLD);

   // receive displacement vector
   rdisp[0] = 0;
   for (int ii = 0; ii < numprocs; ii++)
   {
      rdisp[ii + 1] = rdisp[ii] + rcounts[ii];
   }

   // get the rcounts data
   MPI_Alltoall(rcounts, 1, MPI_INT, scounts, 1, MPI_INT, MPI_COMM_WORLD);

   // get the size of send and receive buffer
   for (int ii = 0; ii < numprocs; ii++)
   {
      ssize += scounts[ii];
      rsize += rcounts[ii];
   }

   // send displacement vector
   sdisp[0] = 0;
   for (int ii = 0; ii < numprocs; ii++)
   {
      sdisp[ii + 1] = sdisp[ii] + scounts[ii];
   }

   MPI_Alltoallv(rlist, rcounts, rdisp, MPI_INT, slist, scounts, sdisp, MPI_INT, MPI_COMM_WORLD);

   // remove offset from slist to get local numbers
   for (int ii = 0; ii < ssize; ii++)
   {
      slist[ii] = slist[ii] - offset;
   }

   // get rlist
   for (int ii = 0; ii < rsize; ii++)
   {
      rlist[ii] = num_solved + ii;
   }

   sbuf = new strict_fp_t[ssize];
   rbuf = new strict_fp_t[rsize];

   mpiparams = new MpiClass();

   mpiparams->ssize = ssize;   
   mpiparams->rsize = rsize;   
   mpiparams->sdisp = sdisp;   
   mpiparams->rdisp = rdisp;   
   mpiparams->scounts = scounts;   
   mpiparams->rcounts = rcounts;
   mpiparams->sbuf = sbuf;
   mpiparams->rbuf = rbuf;
   mpiparams->slist = slist;
   mpiparams->rlist = rlist;

#ifdef ENABLE_GPU
   if(gpu_solver)
   {
      mpiparams->gpu_sbuf = GDF::malloc_gpu_var<strict_fp_t>(ssize);
      mpiparams->gpu_rbuf = GDF::malloc_gpu_var<strict_fp_t>(rsize);
      mpiparams->gpu_slist = GDF::malloc_gpu_var<int>(involved_node);
      mpiparams->gpu_rlist = GDF::malloc_gpu_var<int>(involved_node);
      GDF::memcpy_gpu_var(mpiparams->gpu_sbuf, sbuf, ssize);
      GDF::memcpy_gpu_var(mpiparams->gpu_rbuf, rbuf, rsize);
      GDF::memcpy_gpu_var(mpiparams->gpu_slist, slist, involved_node);
      GDF::memcpy_gpu_var(mpiparams->gpu_rlist, rlist, involved_node);
   }
#endif

   // test mpi transfer
   // set up local vec array
   strict_fp_t *vec = new strict_fp_t[num_involved];
   for (int ii = 0; ii < num_solved; ii++)
   {
      // vec[ii] = ii+offset;
      vec[ii] = gcinvnum[ii + offset];
   }

   mpi_nbnb_transfer(vec);

   for (int ii = num_solved; ii < num_involved; ii++)
   {
      int jj = vec[ii];
      local_global[ii] = jj;  //ghost cells
      global_local[jj] = ii;
   }

   // Calculate num_faces
   for (int i = 0; i < num_solved; i++)
   {
      const int ig = local_global[i];
      num_faces += number_of_neighbors[ig + 1] - number_of_neighbors[ig];
   }

   // Calculate num_attached
   for(int i = 0; i < Nx; i++)
   {
      const int j = 0;
      const int cell = (j * Nx) + i;
      if(ranks_list[cell] == rank)
      {
         num_attached++;
      }
   }
   for(int i = 0; i < Nx; i++)
   {
      const int j = Ny - 1;
      const int cell = (j * Nx) + i;
      if(ranks_list[cell] == rank)
      {
         num_attached++;
      }
   }
   for(int j = 0; j < Ny; j++)
   {
      const int i = 0;
      const int cell = (j * Nx) + i;
      if(ranks_list[cell] == rank)
      {
         num_attached++;
      }
   }
   for(int j = 0; j < Ny; j++)
   {
      const int i = Nx - 1;
      const int cell = (j * Nx) + i;
      if(ranks_list[cell] == rank)
      {
         num_attached++;
      }
   }

   // free the intermediate array
   delete[] offsets_array;
   delete[] involved_array;
   delete[] gcnum;
   delete[] gcinvnum;
   delete[] vec;
}

void mpi_nbnb_transfer(strict_fp_t *vec)
{
   int ssize = mpiparams->ssize;   
   int rsize = mpiparams->rsize;   
   strict_fp_t* sbuf = mpiparams->sbuf;   
   strict_fp_t* rbuf = mpiparams->rbuf;   
   int* slist = mpiparams->slist;   
   int* rlist = mpiparams->rlist;   
   int* sdisp = mpiparams->sdisp;   
   int* rdisp = mpiparams->rdisp;   
   int* scounts = mpiparams->scounts;   
   int* rcounts = mpiparams->rcounts;   

    // get the send buffer
   for(int ii=0;ii<ssize;ii++)
   {
      sbuf[ii] = vec[slist[ii]];
   }

   // communicate
   MPI_Alltoallv(sbuf, scounts, sdisp, MPI_DOUBLE, rbuf, rcounts, rdisp, MPI_DOUBLE, MPI_COMM_WORLD);

   //get the contents from the receive buffer
   for(int ii=0;ii<rsize;ii++)
   {
      vec[rlist[ii]] = rbuf[ii];
   }
}

void Grid::print_grid_information ()
{
   VectorRead<int> number_of_neighbors = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("number_of_neighbors");
   VectorRead<int> cell_neighbors = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("cell_neighbors");
   VectorRead<int> boundary_type_start_and_end_index = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("boundary_type_start_and_end_index");
   VectorRead<strict_fp_t> xcen_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::VECTOR>("xcen_local");
   BoundaryRead<int> boundary_face_to_cell_local = m_silo.retrieve_entry<int, CDF::StorageType::BOUNDARY>("boundary_face_to_cell_local");
   BoundaryRead<strict_fp_t> boundary_area_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::BOUNDARY>("boundary_area_local");
   VectorRead<strict_fp_t> boundary_xcen_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::VECTOR>("boundary_xcen_local");
   VectorRead<strict_fp_t> boundary_normal_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::VECTOR>("boundary_normal_local");

   for (unsigned int i = 0; i < num_solved; i++)
   {
      printf("Cell %d: x %f y %f\n", i, xcen_local[2*i], xcen_local[(2*i) + 1]);
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
   
   for (unsigned int i = 0; i < num_attached; i++)
   {
      printf("Boundary face %d: Cell %d\n", i, boundary_face_to_cell_local[i]);
   }
   
   for (unsigned int i = 0; i < num_attached; i++)
   {
      printf("Boundary face %d: x %f y %f\n", i, boundary_xcen_local[2*i], boundary_xcen_local[(2*i) + 1]);
   }
   
   for (unsigned int i = 0; i < num_attached; i++)
   {
      printf("Boundary face %d: Normal %f %f Area %f\n", i, boundary_normal_local[2*i], 
             boundary_normal_local[2*i+1], boundary_area_local[i]);
   }
}
