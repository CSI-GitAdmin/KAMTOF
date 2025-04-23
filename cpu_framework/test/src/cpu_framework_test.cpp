#include <math.h>

#include "silo.h"
#include "logger.hpp"
#include "silo_fwd.h"
#include "datasetstorage.h"

void foo(loose_fp_t& m_data)
{
   std::cout << "Scaling_factor = " << m_data << std::endl;
}


void register_vars()
{
   Cell<strict_fp_t> pr = m_silo.register_entry<strict_fp_t, CDF::StorageType::CELL>("Pressure");
   Cell<strict_fp_t> vol = m_silo.register_entry<strict_fp_t, CDF::StorageType::CELL>("Volume");
   Cell<strict_fp_t> temp = m_silo.register_entry<strict_fp_t, CDF::StorageType::CELL>("Temperature");
   Parameter<loose_fp_t> scaling_factor = m_silo.register_entry<loose_fp_t, CDF::StorageType::PARAMETER>("Scaling_Factor");

   Vector<strict_fp_t> random_sized_vector = m_silo.register_entry<strict_fp_t, CDF::StorageType::VECTOR>("My_Vec");
   random_sized_vector.resize(100);


   FaceRead<loose_fp_t> tt = m_silo.register_entry<loose_fp_t, CDF::StorageType::FACE>("Temp");

   strict_fp_t init_pr_val(101325.33),init_temp_val(298.15), init_vol_val(1.01);

   scaling_factor = 8.314f;
   for(uint64_t kk = 0; kk < m_silo.get_size<CDF::StorageType::CELL>(); kk++)
   {
      pr[kk] = init_pr_val;
      vol[kk] = init_vol_val;
      temp[kk] = init_temp_val;
   }

   foo(scaling_factor);
}

void test_multidim_data()
{
   const uint8_t polymath_dim = 3;
   const uint8_t polymath_shape[polymath_dim] = {4, 2, 3};

   Cell<int, polymath_dim> cell_polymath_int = m_silo.register_entry<int, CDF::StorageType::CELL, polymath_dim>("Polymath_int", polymath_shape);
   Cell<loose_fp_t, polymath_dim> cell_polymath_dbl = m_silo.register_entry<loose_fp_t, CDF::StorageType::CELL, polymath_dim>("Polymath_dbl", polymath_shape);
   Face<strict_fp_t, polymath_dim> face_polymath_dbl = m_silo.register_entry<strict_fp_t, CDF::StorageType::FACE, polymath_dim>("Polymath_dbl", polymath_shape);

   uint64_t num_cells = m_silo.get_size<CDF::StorageType::CELL>();
   uint64_t num_faces = m_silo.get_size<CDF::StorageType::FACE>();

   for(int cur_cell = 0; cur_cell < num_cells; cur_cell++)
   {
      for(int cur_dim1 = 0; cur_dim1 < polymath_shape[0]; cur_dim1++)
      {
         for(int cur_dim2 = 0; cur_dim2 < polymath_shape[1]; cur_dim2++)
         {
            for(int cur_dim3 = 0; cur_dim3 < polymath_shape[2]; cur_dim3++)
            {
               cell_polymath_int(cur_cell, cur_dim1, cur_dim2, cur_dim3) = cur_cell - cur_dim1 + cur_dim2 - cur_dim3;
               cell_polymath_dbl(cur_cell, cur_dim1, cur_dim2, cur_dim3) = cell_polymath_int(cur_cell, cur_dim1, cur_dim2, cur_dim3);
            }
         }
      }
   }

   for(int cur_cell = 0; cur_cell < num_faces; cur_cell++)
   {
      for(int cur_dim1 = 0; cur_dim1 < polymath_shape[0]; cur_dim1++)
      {
         for(int cur_dim2 = 0; cur_dim2 < polymath_shape[1]; cur_dim2++)
         {
            for(int cur_dim3 = 0; cur_dim3 < polymath_shape[2]; cur_dim3++)
            {
               face_polymath_dbl(cur_cell, cur_dim1, cur_dim2, cur_dim3) = cur_cell - cur_dim1 + cur_dim2 - cur_dim3;
            }
         }
      }
   }

   // face_polymath_dbl.resize(30); This should fail as Resizing induvidual SILO variables is only available for VECTOR StorageType

   if(cell_polymath_int(131,3,1,0) == 129)
   {
      log_msg<CDF::LogLevel::PROGRESS>("Check for multi-dim data for CPU framework passed!");
   }

   std::string msg;

   msg = "cell_polymath_dbl(131,3,1,0) = " + std::to_string(cell_polymath_dbl(131,3,1,0));
   log_msg<CDF::LogLevel::PROGRESS>(msg);

   msg = "face_polymath_dbl(131,3,1,0) = " + std::to_string(face_polymath_dbl(131,3,1,0));
   log_msg<CDF::LogLevel::PROGRESS>(msg);
}

int main(int argc, char** argv)
{

   mpi_init(&argc, &argv);

   log_msg<CDF::LogLevel::PROGRESS>("Hello World!");

   const int nx = 32;
   const int ny = 32;
   const int num_cells = nx * ny;

   m_silo.resize<CDF::StorageType::CELL>(num_cells);
   m_silo.resize<CDF::StorageType::FACE>(num_cells*2);
   m_silo.resize<CDF::StorageType::BOUNDARY>((nx*2) + (ny*2));

   register_vars();

   Cell<strict_fp_t> P = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("Pressure");
   CellRead<strict_fp_t> T = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("Temperature");
   CellRead<strict_fp_t> V = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("Volume");
   ParameterRead<loose_fp_t> alpha = m_silo.retrieve_entry<loose_fp_t, CDF::StorageType::PARAMETER>("Scaling_Factor");

   for(uint64_t kk = 0; kk < num_cells; kk++)
   {
      P[kk] = (alpha * T[kk])/V[kk];
   }

   Vector<strict_fp_t> my_vec = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::VECTOR>("My_Vec");
   my_vec[10] = 8.314;

   strict_fp_t final_pr_val = (298.15 * my_vec[10])/1.01;

   for(uint64_t kk = 0; kk < num_cells; kk++)
   {
      strict_fp_t err = std::fabs(P[kk] - final_pr_val)/P[kk];
      if(err >= 1.0e-04)
      {
         std::string msg = "Incorrect pressure value at index " + std::to_string(kk) + " : P = " + std::to_string(P[kk]) + " || final_pr_val = " + std::to_string(final_pr_val) +
                           " leads to error percentage of " + std::to_string(err);
         log_msg<CDF::LogLevel::ERROR>(msg);
      }
   }

   test_multidim_data();

   my_vec.resize(1000);
   if(my_vec[10] != 8.314)
   {
      log_error("Failed in vector resize operation!");
   }

   BoundaryRead<loose_fp_t> unResolvedVar = m_silo.retrieve_entry<loose_fp_t, CDF::StorageType::BOUNDARY>("Var_which_does_not_exist_yet");
   if(unResolvedVar.exists())
   {
      log_error("SILO NULL failure!");
   }
   Boundary<loose_fp_t> Resolved_var = m_silo.register_entry<loose_fp_t, CDF::StorageType::BOUNDARY>("Var_which_does_not_exist_yet");
   Resolved_var[nx] = 33.33;
   if (unResolvedVar.exists())
   {
      if(unResolvedVar[nx] != Resolved_var[nx])
      {
         log_error("SILO NULL failure!");
      }
   }
   else
   {
      log_error("SILO NULL failure!");
   }

   m_silo.clear_entries();

   mpi_finalize();

   return 0;
}
