#ifndef INDEXDATASET_H
#define INDEXDATASET_H
/**************************************************************************************************
* All rights reserved by CONVERGENT SCIENCE                                                        *
* All information contained herein is the property of Convergent Science.                          *
* The intellectual and technical concepts contained herein are the property of Convergent Science. *
* Dissemination of this information or reproduction of this material is strictly forbidden         *
* unless prior written permission is obtained from Convergent Science.                             *
***************************************************************************************************/
#include <stddef.h>

template <class T, std::size_t M>
struct indexer
{
   template <class... Indices>
   static inline typename std::enable_if<(M == sizeof...(Indices)), T&>::type access(
      T* data, const uint32_t* ds, const short size, Indices&&... idx)
   {
      size_t index      = 0;
      size_t indices[M] = {static_cast<size_t>(idx)...};

      for(short ii = 0; ii < size; ii++)
      {
         index += ds[ii] * indices[ii];
      }
      return data[index];
   }

   template <class... Indices>
   static inline const std::enable_if<(M == sizeof...(Indices)), const T&> access_const(
      const T* data, const uint32_t* ds, const short size, Indices&&... idx)
   {
      size_t index      = 0;
      size_t indices[M] = {static_cast<size_t>(idx)...};

      for(short ii = 0; ii < size; ii++)
      {
         index += ds[ii] * indices[ii];
      }
      return data[index];
   }
};

template <class T>
struct indexer<T,0>
{
   static inline T& access( T* data, const uint32_t* ds, const short size, const size_t& ii)
   {
#ifndef DISABLE_GPU_KERNEL_ASSERTS
      assert(ds[0] == 1);
      assert(size>=1);
#endif
      size_t index = ii /** ds[0]*/;
      return data[index];
   }

   static inline const T& access_const(const T* data, const uint32_t* ds, const short size,
                                       const size_t& ii)
   {
#ifndef DISABLE_GPU_KERNEL_ASSERTS
      assert(ds[0] == 1);
#endif
      size_t index = ii /** ds[0]*/;
      return data[index];
   }
};


template <class T>
struct indexer<T, 1>
{
   static inline T& access(T* data, const uint32_t* ds, const short size, const size_t& ii, const size_t& jj)
   {
#ifndef DISABLE_GPU_KERNEL_ASSERTS
      assert(ds[1] == 1);
      assert(size>=2);
#endif
      size_t index = ii * ds[0] + jj /** ds[1]*/;
      return data[index];
   }

   static inline const T& access_const(const T* data,
                                       const uint32_t* ds, const short size,
                                       const size_t& ii,
                                       const size_t& jj)
   {
#ifndef DISABLE_GPU_KERNEL_ASSERTS
      assert(ds[1] == 1);
      assert(size>=2);
#endif
      size_t index = ii * ds[0] + jj /** ds[1]*/;
      return data[index];
   }
};

template <class T>
struct indexer<T, 2>
{
   static inline T& access(T* data, const uint32_t* ds, const short size, const size_t& ii, const size_t& jj, const size_t& kk)
   {
#ifndef DISABLE_GPU_KERNEL_ASSERTS
      assert(ds[2] == 1);
      assert(size>=3);
#endif
      size_t index = ii * ds[0] + jj * ds[1] + kk /** ds[2]*/;
      return data[index];
   }

   static inline const T& access_const(
      const T* data, const uint32_t* ds, const short size, const size_t& ii, const size_t& jj, const size_t& kk)
   {
#ifndef DISABLE_GPU_KERNEL_ASSERTS
      assert(ds[2] == 1);
#endif
      size_t index = ii * ds[0] + jj * ds[1] + kk /** ds[2]*/;
      return data[index];
   }
};

template <class T>
struct indexer<T, 3>
{
   static inline T& access(
      T* data, const uint32_t* ds, const short size, const size_t& ii, const size_t& jj, const size_t& kk, const size_t& ll)
   {
#ifndef DISABLE_GPU_KERNEL_ASSERTS
      assert(ds[3] == 1);
      assert(size>=4 && ds[4] == 0);
#endif
      size_t index = ii * ds[0] + jj * ds[1] + kk * ds[2] + ll /** ds[3]*/;
      return data[index];
   }

   static inline const T& access_const(
      const T* data, const uint32_t* ds, const short size, const size_t& ii, const size_t& jj, const size_t& kk, const size_t& ll)
   {
#ifndef DISABLE_GPU_KERNEL_ASSERTS
      assert(ds[3] == 1);
#endif
      size_t index = ii * ds[0] + jj * ds[1] + kk * ds[2] + ll /** ds[3]*/;
      return data[index];
   }
};

template <class T>
struct indexer<T, 4>
{
   static inline T& access(
      T* data, const uint32_t* ds, const short size, const size_t& ii, const size_t& jj, const size_t& kk, const size_t& ll, const size_t& mm)
   {
#ifndef DISABLE_GPU_KERNEL_ASSERTS
      assert(size>=5);
      assert(ds[4] == 1);
#endif
      size_t index = ii * ds[0] + jj * ds[1] + kk * ds[2] + ll * ds[3] + mm /** ds[4]*/;
      return data[index];
   }

   static inline const T& access_const(
      const T* data, const uint32_t* ds, const short size, const size_t& ii, const size_t& jj, const size_t& kk, const size_t& ll, const size_t& mm)
   {
#ifndef DISABLE_GPU_KERNEL_ASSERTS
      assert(ds[4] == 1);
#endif
      size_t index = ii * ds[0] + jj * ds[1] + kk * ds[2] + ll * ds[3] + mm /** ds[4]*/;
      return data[index];
   }
};

template <class T>
struct indexer<T, 5>
{
   static inline T& access(
      T* data, const uint32_t* ds, const short size, const size_t& ii, const size_t& jj, const size_t& kk, const size_t& ll, const size_t& mm, const size_t& nn)
   {
#ifndef DISABLE_GPU_KERNEL_ASSERTS
      assert(ds[5] == 1);
      assert(size>=6);
#endif
      size_t index = ii * ds[0] + jj * ds[1] + kk * ds[2] + ll * ds[3] + mm * ds[4] + nn /** ds[5]*/;
      return data[index];
   }

   static inline const T& access_const(
      const T* data, const uint32_t* ds, const short size, const size_t& ii, const size_t& jj, const size_t& kk, const size_t& ll, const size_t& mm, const size_t& nn)
   {
#ifndef DISABLE_GPU_KERNEL_ASSERTS
      assert(ds[5] == 1);
#endif
      size_t index = ii * ds[0] + jj * ds[1] + kk * ds[2] + ll * ds[3] + mm * ds[4] + nn /** ds[5]*/;
      return data[index];
   }
};
#endif // INDEXDATASET_H
