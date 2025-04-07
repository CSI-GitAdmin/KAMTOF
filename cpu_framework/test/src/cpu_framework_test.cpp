#include <math.h>

#include "silo.h"
#include "logger.hpp"
#include "silo_fwd.h"
#include "datasetstorage.h"

void foo(loose_fp_t& m_data)
{
   std::cout << "Scaling_factor = " << m_data << std::endl;
}


void register_vars(silo& m_silo)
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

int main(int argc, char** argv)
{
   log_msg<CDF::LogLevel::PROGRESS>("Hello World!");
   silo m_silo;

   const int nx = 32;
   const int ny = 32;
   const int num_cells = nx * ny;

   m_silo.resize<CDF::StorageType::CELL>(num_cells);
   m_silo.resize<CDF::StorageType::FACE>(num_cells*2);
   m_silo.resize<CDF::StorageType::BOUNDARY>((nx*2) + (ny*2));

   register_vars(m_silo);

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

   return 0;
}
