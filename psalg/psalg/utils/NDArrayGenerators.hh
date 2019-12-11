#ifndef PSALG_NDARRAYGENERATORS_H
#define PSALG_NDARRAYGENERATORS_H

//---------------------------------------------------
// Created on 2019-12-09 by Mikhail Dubrovin
//---------------------------------------------------

/** Usage
 *
 *  #include "psalg/calib/NDArray.hh"
 *  #include "psalg/calib/NDArrayGenerators.hh"
 *
 *  typedef psalg::types::shape_t shape_t; // uint32_t
 *  typedef psalg::types::size_t  size_t;  // uint32_t
 *
 *  float data[] = {1,2,3,4,5,6,7,8,9,10,11,12};
 *  shape_t sh[2] = {3,4};
 *  size_t ndim = 2;
 *
 *  // use external data buffer:
 *  NDArray<float> a(sh, ndim, data);
 *
 *  // set from NDArray or XtcData::Array a
 *  NDArray(a);
 *  
 *  // use internal data buffer:
 *  NDArray<float> b(sh, ndim);
 *  // or set it later
 *  b.set_data_buffer(data) 
 *
 *  // 3: instatiate empty object and initialize it later
 *  NDArray<float> c;
 *  c.set_shape(sh, ndim);    // or alias reshape(sh, ndim)
 *  c.set_shape(str_shape);   // sets _shape and _rank from string
 *  c.set_data_buffer(data);
 *  c.set_ndarray(a);         // where a is NDArray or XtcData::Array
 *
 *  size_t   ndim = a.ndim();
 *  size_t   size = a.size();
 *  shape_t* sh   = a.shape();
 *  T* data       = a.data();
 *
 *  T value = a(1,2);
 *
 *  std::cout << "ostream array: " << a << '\n';
 */

#include <algorithm>  // std::fill
#include <random>     // normal, rand
#include <cstdlib>    // std::rand()

#include "psalg/calib/NDArray.hh"

//#include <cstring>  // memcpy
//#include <iostream> //ostream

//using namespace std;

namespace psalg {

//---------
// fill ndarray with constant
template <typename T>
void fill_ndarray_const(NDArray<T>& a, const T& c) {
  std::fill(a.data(), a.data() + a.size(), c);
}

//---------
// fill ndarray with random values in the range [0,100)
template <typename T>
void fill_ndarray_random(NDArray<T>& a, int range=100) {
  for(T* p=a.data(); p<a.data() + a.size(); ++p) *p = (T)(std::rand() % range);
}

//---------
// fill ndarray with random values in the range [0,100)
template <typename T>
void fill_ndarray_normal(NDArray<T>& a, double mean=0, double stddev=1) {
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(mean,stddev);
  for(T* p=a.data(); p<a.data() + a.size(); ++p) *p = (T)distribution(generator);
}

//---------

  //memcpy(&person_copy, &person, sizeof(person) );
  //T* pbegin = a.data();
  //T* pend = pbegin + a.size() * sizeof(T);

//---------

} // namespace psalg

#endif // PSALG_NDARRAYGENERATORS_H
