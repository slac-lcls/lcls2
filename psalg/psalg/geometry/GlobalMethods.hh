#ifndef PSALG_GLOBALMETHODS_H
#define PSALG_GLOBALMETHODS_H
//-------------------

#include <iostream>
#include <string>

//#include "ndarray/ndarray.h"
#include "psalg/calib/NDArray.hh"

//#include <cstddef>  // for size_t

//-------------------

namespace geometry {

/// @addtogroup geometry geometry
/**
 *  @ingroup geometry
 *
 *  @brief module GlobalMethods.h has Global Methods
 *
 */

//-------------------

using namespace psalg;

typedef psalg::types::shape_t shape_t;
typedef psalg::types::size_t  size_t;
  //typedef psalg::NDArray NDArray;

//-------------------

static const size_t N2X1    = 2;
static const size_t ROWS2X1 = 185;
static const size_t COLS2X1 = 388;
static const size_t SIZE2X1 = COLS2X1*ROWS2X1; 
static const size_t SIZE2X2 = N2X1*SIZE2X1; 

//-------------------

/**
 * @brief Converts cspad2x2 NDArray data2x2[185,388,2] to two2x1[2,185,388] 
 * 
 * @param[in]  data2x2 - input NDArray shaped as [185,388,2]
 */
  template <typename T>
  NDArray<const T> 
  data2x2ToTwo2x1(const NDArray<const T>& data2x2)
  {
    shape_t sh[3] = {N2X1, ROWS2X1, COLS2X1};
    NDArray<T> two2x1 = new NDArray<T>(sh, 3);
    
    for(size_t n=0; n<N2X1;    ++n) {
    for(size_t c=0; c<COLS2X1; ++c) {
    for(size_t r=0; r<ROWS2X1; ++r) {

      two2x1(n,r,c) = data2x2(r,c,n);  

    }
    }
    }
    return *two2x1;
  }

//-------------------
/**
 * @brief Converts cspad2x2 NDArray two2x1[2,185,388] to data2x2[185,388,2]
 * 
 * @param[in]  two2x1 - input NDArray shaped as [2,185,388]
 */
  template <typename T>
  NDArray<const T> 
  two2x1ToData2x2(const NDArray<const T>& two2x1)
  {
    shape_t sh[3] = {N2X1, ROWS2X1, COLS2X1};
    NDArray<T> data2x2 = new NDArray<T>(sh, 3);
    
    for(size_t n=0; n<N2X1;    ++n) {
    for(size_t c=0; c<COLS2X1; ++c) {
    for(size_t r=0; r<ROWS2X1; ++r) {

      data2x2(r,c,n) = two2x1(n,r,c);  

    }
    }
    }
    return *data2x2;
  }

//-------------------
/**
 * @brief Converts cspad2x2 NDArray two2x1[2,185,388] to data2x2[185,388,2]
 * 
 * @param[in]  A - pointer to input array with data shaped as [2,185,388]
 */
  template <typename T>
  void two2x1ToData2x2(T* A)
  {
    shape_t shape_in [3] = {N2X1, ROWS2X1, COLS2X1};
    shape_t shape_out[3] = {ROWS2X1, COLS2X1, N2X1};

    NDArray<T> two2x1(shape_in, 3, A);
    NDArray<T> data2x2(shape_out, 3);

    for(size_t n=0; n<N2X1;    ++n) {
    for(size_t c=0; c<COLS2X1; ++c) {
    for(size_t r=0; r<ROWS2X1; ++r) {

      data2x2(r,c,n) = two2x1(n,r,c);

    }
    }
    }
    std::memcpy(A, data2x2.data(), data2x2.size()*sizeof(T));
  }

} // namespace geometry
//-------------------

#endif // PSALG_GLOBALMETHODS_H
