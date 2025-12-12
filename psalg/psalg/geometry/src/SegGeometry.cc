//-------------------

#include "psalg/geometry/SegGeometry.hh"

#include <math.h>      // sin, cos
//#include <iostream>    // cout

namespace geometry {

SegGeometry::SegGeometry(){}

SegGeometry::~SegGeometry(){}

//--------------

template <typename T>
T min_of_arr(const T* arr, gsize_t size)
{
  T min=arr[0]; for(gsize_t i=1; i<size; ++i) {if(arr[i] < min) min=arr[i];} 
  return min;
}

//--------------

template <typename T>
T max_of_arr(const T* arr, gsize_t size)
{
  T max=arr[0]; for(gsize_t i=1; i<size; ++i) {if(arr[i] > max) max=arr[i];} 
  return max;
}

//----------------

template double min_of_arr<double>(const double*, gsize_t);
template float  min_of_arr<float> (const float*,  gsize_t);
template int    min_of_arr<int>   (const int*,    gsize_t);

template double max_of_arr<double>(const double*, gsize_t);
template float  max_of_arr<float> (const float*,  gsize_t);
template int    max_of_arr<int>   (const int*,    gsize_t);

//--------------

} // namespace geometry

//--------------
