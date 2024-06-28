#ifndef PSALGOS_LOCALEXTREMA_H
#define PSALGOS_LOCALEXTREMA_H

//-----------------------------
// LocalExtrema.h 2017-08-04
//-----------------------------

#include <string>
#include <vector>
#include <iostream> // for cout, ostream
#include <cstddef>  // for size_t
#include <cstring>  // for memcpy
#include <cmath>    // for sqrt

#include "Types.hh"
#include "psalg/alloc/AllocArray.hh"
#include "psalg/alloc/Allocator.hh"

//-----------------------------

using namespace std;
using namespace psalg; // Array

//-----------------------------

namespace localextrema {

/**
 *  @ingroup psalgos
 *
 *  @brief LocalExtrema - methods for 2-d image processing algorithms.
 *
 *  This software was developed for the LCLS project.  
 *  If you use all or part of it, please give an appropriate acknowledgment.
 *
 *  @author Mikhail Dubrovin
 *
 *  @see ImgAlgos.ImgImgProc
 *
 *  @anchor interface
 *  @par<interface> Interface Description
 *
 * 
 *  @li  Includes and typedefs
 *  @code
 *  #include <cstddef>  // for size_t
 *  #include "psalgos/LocalExtrema.h"
 *  #include "psalgos/Types.h"
 *
 *  typedef types::mask_t     mask_t;
 *  typedef types::extrim_t   extrim_t;
 *  @endcode
 *
 *
 *  @li Define input parameters
 *  \n
 *  @code
 *    const T *data = ...
 *    const mask_t *mask = ...
 *    const size_t rows = 1000;
 *    const size_t cols = 1000;
 *    const size_t rank = 5;
 *    const T thr_low = 30;
 *    const T thr_high = 60;
 *    extrim_t *local_maxima = new extrim_t[rows*cols];
 *  @endcode
 *
 *
 *  @li Call methods
 *  \n
 *  @code
 *  size_t n = localMinima1d(data, mask, cols, stride, rank, local_minima);
 *  size_t n = localMaxima1d(data, mask, cols, stride, rank, local_maxima);
 *  
 *  size_t n = mapOfLocalMinimums(data, mask, rows, cols, rank, arr2d);
 *  size_t n = mapOfLocalMaximums(data, mask, rows, cols, rank, arr2d);
 *  size_t n = mapOfLocalMaximumsRank1Cross(data, mask, rows, cols, arr2d);
 *  size_t n = mapOfThresholdMaximums(data ,mask, rows, cols, rank, thr_low, thr_high, local_maxima)
 *
 *  std::vector<TwoIndexes> v = evaluateDiagIndexes(const size_t& rank);
 *  printMatrixOfDiagIndexes(rank);
 *  printVectorOfDiagIndexes(rank);
 *
 *  const extrim_t vsel=7;
 *  const size_t maxlen=1000;
 *  std::vector<TwoIndexes> v = vectorOfExtremeIndexes(map, rows, cols, vsel, maxlen);
 *  size_t n = numberOfExtrema(local_maxima, rows, cols, vsel);
 *  @endcode
 */

//-----------------------------

typedef types::mask_t     mask_t;
typedef types::extrim_t   extrim_t;
typedef types::TwoIndexes TwoIndexes;

//-----------------------------

AllocArray1D<TwoIndexes> evaluateDiagIndexes_drp(const size_t& rank, Allocator *allocator);
void printMatrixOfDiagIndexes(const size_t& rank);
void printVectorOfDiagIndexes(const size_t& rank);
size_t numberOfExtrema(const extrim_t *map, const size_t& rows, const size_t& cols, const extrim_t& vsel=7);
std::vector<TwoIndexes> evaluateDiagIndexes(const size_t& rank);
std::vector<TwoIndexes> vectorOfExtremeIndexes(const extrim_t *map, const size_t& rows, const size_t& cols, const extrim_t& vsel=7, const size_t& maxlen=0);

//-----------------------------
  /**
   * @brief returns number of found minima and array local_minima of local minimums of requested rank, 
   *        where rank defins a square region [cols-rank, cols+rank]. Assumes that cols > 2*rank...
   * 
   * 1-d array of local minumum of (uint16) values of data shape, 
   * with 0/+1/+2 for bad/non-minumum/minimum in rank region.   
   * @param[in]  data - pointer to data array
   * @param[in]  mask - pointer to mask array; mask marks bad/good (0/1) pixels
   * @param[in]  cols - number of columns in 1d array
   * @param[in]  stride - index increment between neighbour elements. Makes sence for n-d arrays
   * @param[in]  rank - radius of the square region in which central pixel has a maximal value
   * @param[out] local_minima  - pointer to the array local_minima
   */
//-----------------------------

template <typename T>
size_t 
localMinima1d(const T *data
             ,const mask_t *mask
             ,const size_t& cols
             ,const size_t& stride=1
             ,const size_t& rank=5
             ,extrim_t *local_minima=0
             )
{
  #ifndef NDEBUG
  std::cout << "in localMinima1d, rank=" << rank << "\n";
  #endif

  extrim_t *_local_minima = local_minima;
  size_t counter = 0;

  for(unsigned c=0, i=0; c<cols; c++, i+=stride) _local_minima[i]=(mask[i]) ? 1 : 0; // good/bad pixel

  unsigned i=0, ii=0;

  // check for extrema at the low edge [0,runk).
  for(unsigned c=0; c<rank; c++) {
      i = c*stride;
      if(!mask[i]) continue;
      if(!(data[i]<data[i+stride])) continue;
      _local_minima[i] |= 2;
      for(unsigned cc=c+1; cc<c+rank+1; cc++) {
      ii = cc*stride;
      if(mask[ii] && (data[ii] < data[i])) {
          _local_minima[i] &=~2; // clear 2nd bit
          c = cc - 1;            // jump ahead, c will be incremented in the for loop
          break;
      }
      }
      if(_local_minima[c] & 2) {
          counter ++;
          break;
      }
  }

  // check for extreme in the range [rank, cols-rank)
  for(unsigned c=rank; c<cols-rank; c++) {
      i = c*stride;
      if(!mask[i]) continue;
      if(!(data[i]<data[i+stride])) continue;
      _local_minima[i] |= 2; // set 2nd bit

      // check positive side of c
      for(unsigned cc=c+1; cc<c+rank+1; cc++) {
      ii = cc*stride;
      if(mask[ii] && (data[ii] < data[i])) {
              _local_minima[i] &=~2; // clear 2nd bit
              c = cc - 1;            // jump ahead, c will be incremented in the for loop 
          break;
      }
      }

      if(_local_minima[i] & 2) {
          // check negative side of c
      for(unsigned cc=c-rank; cc<c; cc++) {
          ii = cc*stride;
          if(mask[ii] && (data[ii] < data[i])) {
              _local_minima[i] &=~2; // clear 2nd bit
              c = cc + rank;  // jump ahead, c will be incremented in the for loop
              break;
          }
          }
      }
      if(_local_minima[c] & 2) counter ++;
  } // loop in the range [rank, cols-rank]

  // check for extreme at the high edge [cols-rank, cols).
  for(unsigned c=cols-1; c>cols-rank-1; c--) {
      i = c*stride;
      if(!mask[i]) continue;
      if(!(data[i]<data[i-stride])) continue;
      _local_minima[i] |= 2;
      for(unsigned cc=c-1; cc>c-rank-1; cc--) {
      ii = cc*stride;
      if(mask[ii] && (data[ii] < data[i])) {
          _local_minima[i] &=~2; // clear 2nd bit
          c = cc;                // jump ahead, c will be incremented in the for loop
          break;
      }
      }
      if(_local_minima[c] & 2) {
          counter ++;
          break;
      }

  }
  return counter;
}

//-----------------------------
//-----------------------------
  /**
   * @brief returns number of found maxima and array local_maxima of local maximums of requested rank, 
   *        where rank defins a region [cols-rank, cols+rank]. Assumes that cols > 2*rank...
   * 
   * 1-d array of local maxumum of (uint16) values of data shape, 
   * with 0/+1/+2 for bad/non-maxumum/maximum in rank region.   
   * @param[in]  data - pointer to data array
   * @param[in]  mask - pointer to mask array; mask marks bad/good (0/1) pixels
   * @param[in]  cols - number of columns in 1d array
   * @param[in]  stride - index increment between neighbour elements. Makes sence for n-d arrays
   * @param[in]  rank - radius of the square region in which central pixel has a maximal value
   * @param[out] local_maxima  - pointer to the array local_maxima
   */
//-----------------------------

template <typename T>
size_t 
localMaxima1d(const T *data
             ,const mask_t *mask
             ,const size_t& cols
             ,const size_t& stride=1
             ,const size_t& rank=5
             ,extrim_t *local_maxima=0
             )
{
  #ifndef NDEBUG
  std::cout << "in localMaxima1d, rank=" << rank << "\n";
  #endif

  extrim_t *_local_maxima = local_maxima;
  size_t counter = 0;

  for(unsigned c=0, i=0; c<cols; c++, i+=stride) _local_maxima[i]=(mask[i]) ? 1 : 0; // good/bad pixel

  unsigned i=0, ii=0;

  // check for extrema at the low edge [0,runk).
  for(unsigned c=0; c<rank; c++) {
      i = c*stride;
      if(!mask[i]) continue;
      if(!(data[i]>data[i+stride])) continue;
      _local_maxima[i] |= 2;
      for(unsigned cc=c+1; cc<c+rank+1; cc++) {
      ii = cc*stride;
      if(mask[ii] && (data[ii] > data[i])) {
          _local_maxima[i] &=~2; // clear 2nd bit
          c = cc - 1;            // jump ahead, c will be incremented in the for loop
          break;
      }
      }
      if(_local_maxima[c] & 2) {
          counter ++;
          break;
      }
  }

  // check for extreme in the range [rank, cols-rank)
  for(unsigned c=rank; c<cols-rank; c++) {
      i = c*stride;
      if(!mask[i]) continue;
      if(!(data[i]>data[i+stride])) continue;
      _local_maxima[i] |= 2; // set 2nd bit

      // check positive side of c
      for(unsigned cc=c+1; cc<c+rank+1; cc++) {
      ii = cc*stride;
      if(mask[ii] && (data[ii] > data[i])) {
          _local_maxima[i] &=~2; // clear 2nd bit
          c = cc - 1;            // jump ahead, c will be incremented in the for loop
          break;
      }
      }

      if(_local_maxima[i] & 2) {
          // check negative side of c
      for(unsigned cc=c-rank; cc<c; cc++) {
          ii = cc*stride;
          if(mask[ii] && (data[ii] > data[i])) {
              _local_maxima[i] &=~2; // clear 2nd bit
              c = cc + rank;  // jump ahead, c will be incremented in the for loop
              break;
          }
          }
      }
      if(_local_maxima[c] & 2) counter ++;
  } // loop in the range [rank, cols-rank]

  // check for extreme at the high edge [cols-rank, cols).
  for(unsigned c=cols-1; c>cols-rank-1; c--) {
      i = c*stride;
      if(!mask[i]) continue;
      if(!(data[i]>data[i-stride])) continue;
      _local_maxima[i] |= 2;
      for(unsigned cc=c-1; cc>c-rank-1; cc--) {
      ii = cc*stride;
      if(mask[ii] && (data[ii] > data[i])) {
          _local_maxima[i] &=~2; // clear 2nd bit
          c = cc;                // jump ahead, c will be incremented in the for loop
          break;
      }
      }
      if(_local_maxima[c] & 2) {
          counter ++;
          break;
      }

  }
  return counter;
}

//-----------------------------

//-----------------------------
  /**
   * @brief returns map of local minimums of requested rank, 
   *        where rank defins a square region around central pixel [rowc-rank, rowc+rank], [colc-rank, colc+rank].
   * 
   * Map of local minumum is a 2-d array of (uint16) values of data shape, 
   * with 0/+1/+2/+4 for non-minumum / minumum in column / minumum in row / minimum in square of radius rank.   
   * @param[in]  data - pointer to data array
   * @param[in]  mask - pointer to mask array; mask marks bad/good (0/1) pixels
   * @param[in]  rows - number of rows in all 2-d arrays
   * @param[in]  cols - number of columns in all 2-d arrays
   * @param[in]  rank - radius of the square region in which central pixel has a maximal value
   * @param[out] local_minima  - pointer to map of local minimums
   */
//-----------------------------

template <typename T>
size_t 
mapOfLocalMinimums_drp(const T *data
                  ,const mask_t *mask
                  ,const size_t& rows
                  ,const size_t& cols
                  ,const size_t& rank
                  ,extrim_t *local_minima
                  ,Allocator *allocator
                  )
{
  #ifndef NDEBUG
  std::cout << "XXX mapOfLocalMinimums point E \n";
  #endif

  // initialization of indexes
  AllocArray1D<TwoIndexes> arr_inddiag_drp = evaluateDiagIndexes_drp(rank, allocator);

  extrim_t *_local_minima = local_minima;
  size_t size = rows*cols;
  std::fill_n(_local_minima, int(size), extrim_t(0));

  size_t counter = 0;

  unsigned rmin = 0;
  unsigned rmax = (int)rows;
  unsigned cmin = 0;
  unsigned cmax = (int)cols;
  int irank = (int)rank;

  int irc=0;
  int ircd=0;
  int irdc=0;
  int iric=0;

  // check rank minimum in columns and set the 1st bit (1)
  for(unsigned r = rmin; r<rmax; r++) {
    for(unsigned c = cmin; c<cmax; c++) {
      irc = r*cols+c;

      if(!mask[irc]) continue;
      if((c+1<cmax) && !(data[irc]<data[irc+1])) continue;
      _local_minima[irc] = 1;

      // positive side of c 
      unsigned dmax = min((int)cmax-1, int(c)+irank);
      for(unsigned cd=c+1; cd<=dmax; cd++) {
        ircd = r*cols+cd;
        if(mask[ircd] && (data[ircd] < data[irc])) {
              _local_minima[irc] &=~1; // clear 1st bit
              c=cd-1; // jump ahead
          break;
        }
      }

      if(_local_minima[irc] & 1) {
        // negative side of c 
        unsigned dmin = max((int)cmin, int(c)-irank);
        for(unsigned cd=dmin; cd<c; cd++) {
          ircd = r*cols+cd;
          if(mask[ircd] && (data[ircd] < data[irc])) {
                _local_minima[irc] &=~1; // clear 1st bit
                c=cd+rank; // jump ahead
            break;
          }
        }
      }

      // (r,c) is a local dip, jump ahead through the tested rank range
      if(!(c<cmax)) break;
      if(_local_minima[irc] & 1) c+=rank;
    }
  }

  // check rank minimum in rows and set the 2nd bit (2)
  for(unsigned c = cmin; c<cmax; c++) {
    for(unsigned r = rmin; r<rmax; r++) {
      // if it is not a local maximum from previous algorithm
      //if(!_local_minima[irc]) continue;
      irc = r*cols+c;

      if(!mask[irc]) continue;
      if((r+1<rmax) && !(data[irc]<data[irc+cols])) continue;
      _local_minima[irc] |= 2; // set 2nd bit

      // positive side of r 
      unsigned dmax = min((int)rmax-1, int(r)+irank);
      for(unsigned rd=r+1; rd<=dmax; rd++) {
        irdc = rd*cols+c;
        if(mask[irdc] && (data[irdc] < data[irc])) {
              _local_minima[irc] &=~2; // clear 2nd bit
              r=rd-1; // jump ahead
          break;
        }
      }

      if(_local_minima[irc] & 2) {
        // negative side of r
        unsigned dmin = max((int)rmin, int(r)-irank);
        for(unsigned rd=dmin; rd<r; rd++) {
          irdc = rd*cols+c;
          if(mask[irdc] && (data[irdc] < data[irc])) {
                _local_minima[irc] &=~2; // clear 2nd bit
                r=rd+rank; // jump ahead
            break;
          }
        }
      }

      // (r,c) is a local dip, jump ahead through the tested rank range
      if(!(r<rmax)) break;
      if(_local_minima[irc] & 2) r+=rank;
    }
  }

  // check rank minimum in "diagonal" regions and set the 3rd bit (4)
  for(unsigned r = rmin; r<rmax; r++) {
    for(unsigned c = cmin; c<cmax; c++) {
      irc = r*cols+c;
      // if it is not a local minimum from two previous algorithm
      if(_local_minima[irc] != 3) continue;
      _local_minima[irc] |= 4; // set 3rd bit

      for(unsigned int ii = 0; ii < arr_inddiag_drp.num_elem(); ii++) {
        
        int ir = r + (arr_inddiag_drp(ii).i);
        int ic = c + (arr_inddiag_drp(ii).j);
        
        if(  ir<(int)rmin)  continue;
        if(  ic<(int)cmin)  continue;
        if(!(ir<(int)rmax)) continue;
        if(!(ic<(int)cmax)) continue;

        iric = ir*cols+ic;
        if(mask[iric] && (data[iric] < data[irc])) {
              _local_minima[irc] &=~4; // clear 3rd bit
          break;
        }
      }

      // (r,c) is a local peak, jump ahead through the tested rank range
      if(_local_minima[irc] & 4) {
         c+=rank;
         counter ++;
      }
    }
  }

  return counter;
}

//--------------------
  /**
   * @brief returns map of local maximums of requested rank, 
   *        where rank defins a square region around central pixel [rowc-rank, rowc+rank], [colc-rank, colc+rank].
   * 
   * Map of local maximum is a 2-d array of (uint16) values of data shape, 
   * with 0/+1/+2/+4 for non-maximum / maximum in column / maximum in row / minimum in square of radius rank.   
   * @param[in]  data - pointer to data array
   * @param[in]  mask - pointer to mask array; mask marks bad/good (0/1) pixels
   * @param[in]  rows - number of rows in all 2-d arrays
   * @param[in]  cols - number of columns in all 2-d arrays
   * @param[in]  rank - radius of the square region in which central pixel has a maximal value
   * @param[out] local_maxima  - pointer to map of local maximums
   */

template <typename T>
size_t 
mapOfLocalMaximums_drp(const T *data
                  ,const mask_t *mask
                  ,const size_t& rows
                  ,const size_t& cols
                  ,const size_t& rank
                  ,extrim_t *local_maxima
                  ,Allocator *allocator
                  )
{
  #ifndef NDEBUG
  std::cout << "in mapOfLocalMaximums, rank=" << rank << "\n";
  #endif

  AllocArray1D<TwoIndexes> arr_inddiag_drp = evaluateDiagIndexes_drp(rank, allocator);

  extrim_t *_local_maxima = local_maxima;
  std::fill_n(&_local_maxima[0], int(rows*cols), extrim_t(0));

  size_t counter = 0;

  unsigned rmin = 0;
  unsigned rmax = (int)rows;
  unsigned cmin = 0;
  unsigned cmax = (int)cols;
  int irank = (int)rank;

  int irc=0;
  int ircd=0;
  int irdc=0;
  int iric=0;

  // check rank maximum in columns and set the 1st bit (1)
  for(unsigned r = rmin; r<rmax; r++) {
    for(unsigned c = cmin; c<cmax; c++) {
      irc = r*cols+c;
      if(!mask[irc]) continue;
      if((c+1<cmax) && !(data[irc]>data[irc+1])) continue;
      _local_maxima[irc] = 1;

      // positive side of c 
      unsigned dmax = min((int)cmax-1, int(c)+irank);
      for(unsigned cd=c+1; cd<=dmax; cd++) {
        ircd = r*cols+cd;
        if(mask[ircd] && (data[ircd] > data[irc])) {
              _local_maxima[irc] &=~1; // clear 1st bit
              c=cd-1; // jump ahead
          break;
        }
      }

      if(_local_maxima[irc] & 1) {
        // negative side of c 
        unsigned dmin = max((int)cmin, int(c)-irank);
        for(unsigned cd=dmin; cd<c; cd++) {
          ircd = r*cols+cd;
          if(mask[ircd] && (data[ircd] > data[irc])) {
                _local_maxima[irc] &=~1; // clear 1st bit
                c=cd+rank; // jump ahead
            break;
          }
        }
      }

      // (r,c) is a local dip, jump ahead through the tested rank range
      if(!(c<cmax)) break;
      if(_local_maxima[irc] & 1) c+=rank;
    }
  }

  // check rank maximum in rows and set the 2nd bit (2)
  for(unsigned c = cmin; c<cmax; c++) {
    for(unsigned r = rmin; r<rmax; r++) {
      irc = r*cols+c;
      // if it is not a local maximum from previous algorithm
      //if(!_local_maxima[irc]) continue;

      if(!mask[irc]) continue;
      if((r+1<rmax) && !(data[irc]>data[irc+cols])) continue;
      _local_maxima[irc] |= 2; // set 2nd bit

      // positive side of r 
      unsigned dmax = min((int)rmax-1, int(r)+irank);
      for(unsigned rd=r+1; rd<=dmax; rd++) {
        irdc = rd*cols+c;
        if(mask[irdc] && (data[irdc] > data[irc])) {
              _local_maxima[irc] &=~2; // clear 2nd bit
              r=rd-1; // jump ahead
          break;
        }
      }

      if(_local_maxima[irc] & 2) {
        // negative side of r
        unsigned dmin = max((int)rmin, int(r)-irank);
        for(unsigned rd=dmin; rd<r; rd++) {
          irdc = rd*cols+c;
          if(mask[irdc] && (data[irdc] > data[irc])) {
                _local_maxima[irc] &=~2; // clear 2nd bit
                r=rd+rank; // jump ahead
            break;
          }
        }
      }

      // (r,c) is a local dip, jump ahead through the tested rank range
      if(!(r<rmax)) break;
      if(_local_maxima[irc] & 2) r+=rank;
    }
  }

  // check rank maximum in "diagonal" regions and set the 3rd bit (4)
  for(unsigned r = rmin; r<rmax; r++) {
    for(unsigned c = cmin; c<cmax; c++) {
      // if it is not a local maximum from two previous algorithm
      irc = r*cols+c;

      if(_local_maxima[irc] != 3) continue;
      _local_maxima[irc] |= 4; // set 3rd bit

      for(unsigned int ii = 0; ii < arr_inddiag_drp.num_elem(); ii++) {
        int ir = r + (arr_inddiag_drp(ii).i);
        int ic = c + (arr_inddiag_drp(ii).j);

        if(  ir<(int)rmin)  continue;
        if(  ic<(int)cmin)  continue;
        if(!(ir<(int)rmax)) continue;
        if(!(ic<(int)cmax)) continue;

        iric = ir*cols+ic;
        if(mask[iric] && (data[iric] > data[irc])) {
              _local_maxima[irc] &=~4; // clear 3rd bit
          break;
        }
      }

      // (r,c) is a local peak, jump ahead through the tested rank range
      if(_local_maxima[irc] & 4) {
        c+=rank;
        counter ++;
      }
    }
  }

  return counter;
}
//-----------------------------

  /**
   * @brief returns map of local maximums of runk=1 cross(+) region (very special case for Chuck's algorithm).
   * 
   * Map of local maximum is a 2-d array of (uint16) values of data shape, 
   * with 0/+1/+2 for non-maximum / maximum in column / maximum in row, then local maximum in cross = 3.   
   * @param[in]  data - pointer to data array
   * @param[in]  mask - pointer to mask array; mask marks bad/good (0/1) pixels
   * @param[in]  rows - number of rows in all 2-d arrays
   * @param[in]  cols - number of columns in all 2-d arrays
   * @param[out] local_maxima - pointer to map of local maximums
   */

template <typename T>
size_t 
mapOfLocalMaximumsRank1Cross(const T *data
                            ,const mask_t *mask
                            ,const size_t& rows
                            ,const size_t& cols
                            ,extrim_t *local_maxima
                            )
{
  #ifndef NDEBUG
  std::cout << "in mapOfLocalMaximumsRank1Cross\n";
  #endif

  extrim_t *_local_maxima = local_maxima;
  std::fill_n(&_local_maxima[0], int(rows*cols), extrim_t(0));

  size_t counter = 0;

  unsigned rmin = 0;
  unsigned rmax = (int)rows;
  unsigned cmin = 0;
  unsigned cmax = (int)cols;

  int irc=0;
  int ircm=0;
  int ircm2=0;

  // check local maximum in columns and set the 1st bit (1)
  for(unsigned r = rmin; r<rmax; r++) {

    // first pixel in the row
    unsigned c = cmin;
    irc = r*cols+c;
    if(mask[irc] && mask[irc+1] && (data[irc] > data[irc+1])) {
      _local_maxima[irc] |= 1;  // set 1st bit
      c+=2;
    }
    else c+=1;

    // all internal pixels in the row
    for(; c<cmax-1; c++) {
      irc = r*cols+c;
      if(!mask[irc]) continue;                                       // go to the next pixel
      if(mask[irc+1] && (data[irc+1] >= data[irc])) continue;         // go to the next pixel
      if(mask[irc-1] && (data[irc-1] >= data[irc])) {c+=1; continue;} // jump ahead 
      _local_maxima[irc] |= 1;  // set 1st bit
      c+=1; // jump ahead 
    }

    // last pixel in the row
    ircm = r*cols+cmax-1;
    if(mask[ircm] && mask[ircm-1] && (data[ircm] > data[ircm-1])) _local_maxima[ircm] |= 1;  // set 1st bit
  } // rows loop

  // check local maximum in rows and set the 2nd bit (2)
  for(unsigned c = cmin; c<cmax; c++) {

    // first pixel in the column
    unsigned r = rmin;
    irc  = r*cols+c;
    ircm = (r+1)*cols+c;
    if(mask[irc] && mask[ircm] && (data[irc] > data[ircm])) {
      _local_maxima[irc] |= 2; // set 2nd bit
      r+=2;
    }
    else r+=1;

    // all internal pixels in the column
    for(; r<rmax-1; r++) {
      irc = r*cols+c;
      if(!mask[irc]) continue;
      ircm = (r+1)*cols+c;
      if(mask[ircm] && (data[ircm] >= data[irc])) continue;         // go to the next pixel
      ircm = (r-1)*cols+c;
      if(mask[ircm] && (data[ircm] >= data[irc])) {r+=1; continue;} // jump ahead 
      _local_maxima[irc] |= 2; // set 2nd bit
      r+=1; // jump ahead 

      if(_local_maxima[irc] == 3) counter++;
    }

    // last pixel in the column
    ircm  = (rmax-1)*cols+c;
    ircm2 = (rmax-2)*cols+c;
    if(mask[ircm] && mask[ircm2] && (data[ircm] > data[ircm2])) _local_maxima[ircm] |= 2;  // set 2nd bit
    if(_local_maxima[ircm] == 3) counter++;
  } // columns loop

  return counter;
}

//-----------------------------

  /**
   * @brief returns map of local maximums for 2-threshold algorithm.
   * 
   * Map of local maximum is a 2-d array of (uint16) values of data shape, 
   * with 0/+1/+2/+4/+8 for masked / <thr_low / >thr_low && <thr_high / >thr_high / local maxima in runk-square
   *      +16 < -thr_low
   * @param[in]  data - pointer to data array
   * @param[in]  mask - pointer to mask array; mask marks bad/good (0/1) pixels
   * @param[in]  rows - number of rows in all 2-d arrays
   * @param[in]  cols - number of columns in all 2-d arrays
   * @param[in]  rank - radius of the square region in which central pixel has a maximal value
   * @param[in]  thr_low - low threshold on intensity
   * @param[in]  thr_high - high threshold on intensity
   * @param[out] map  - pointer to map of local maximums
   */

template <typename T>
size_t
mapOfThresholdMaximums(const T *data
		      ,const mask_t *mask
		      ,const size_t& rows
		      ,const size_t& cols
		      ,const size_t& rank
		      ,const double& thr_low
		      ,const T& thr_high
		      ,extrim_t *thr_maxima
		      )
{
  #ifndef NDEBUG
  std::cout << "XXX: in mapOfThresholdMaximums\n" ;
  #endif

  const T _thr_low = thr_low;
  const T _thr_high = thr_high;
  extrim_t *_thr_maxima = thr_maxima;
  std::fill_n(&_thr_maxima[0], int(rows*cols), extrim_t(0));

  size_t counter = 0;

  unsigned rmin = 0;
  unsigned rmax = (int)rows;
  unsigned cmin = 0;
  unsigned cmax = (int)cols;

  int irc=0;
  int rrc=0;
 
  for(unsigned r = rmin; r<rmax; r++) {
    for(unsigned c = cmin; c<cmax; c++) {
      irc = r*cols+c;
      if(! mask[irc])                 _thr_maxima[irc]  = 0;  // pixel is masked
      else if(data[irc] < -_thr_high) _thr_maxima[irc] |= 32; // a<-thr_high
      else if(data[irc] < -_thr_low)  _thr_maxima[irc] |= 16; // a<-thr_low
      else if(data[irc] <  _thr_low)  _thr_maxima[irc] |= 1;  // a<thr_low - for background
      else if(data[irc] <  _thr_high) _thr_maxima[irc] |= 2;  // a>=thr_low, but a<thr_high
      else {                          _thr_maxima[irc] |= 12; // +4: a>=thr_high, +8: candidate to local maximum 

        //----- check if pixel has maximal intensity in rank-square region
	int rrmin = max(0,         int(r-rank));
	int rrmax = min(int(rmax), int(r+rank+1));
	int rcmin = max(0,         int(c-rank));
	int rcmax = min(int(cmax), int(c+rank+1));

        //cout << "  candidate r:" << r << " c:" << c << " map[irc]:" << _thr_maxima[irc]  << " data:" << data[irc] << '\n';

        counter ++;
        for(int rr = rrmin; rr<rrmax; rr++) {
          for(int rc = rcmin; rc<rcmax; rc++) {
            rrc  = rr*cols+rc;
            if(rrc==irc or !mask[rrc] or _thr_maxima[rrc]==4) continue; 
            if(!(data[irc]>data[rrc])) {
              _thr_maxima[irc] = 4; // a>=thr_high, but NOT a local maximum in rank
              //_thr_maxima[irc] &= ~8; // a>=thr_high, but NOT a local maximum in rank

              counter --;
              rr = rrmax; // termenate rr loop
              break;      // termenate rc loop
            }
          }
        }
        //-----
      }
    }
  }
  return counter;
}

//-----------------------------



//-----------------------------
  /**
   * @brief returns map of local minimums of requested rank, 
   *        where rank defins a square region around central pixel [rowc-rank, rowc+rank], [colc-rank, colc+rank].
   * 
   * Map of local minumum is a 2-d array of (uint16) values of data shape, 
   * with 0/+1/+2/+4 for non-minumum / minumum in column / minumum in row / minimum in square of radius rank.   
   * @param[in]  data - pointer to data array
   * @param[in]  mask - pointer to mask array; mask marks bad/good (0/1) pixels
   * @param[in]  rows - number of rows in all 2-d arrays
   * @param[in]  cols - number of columns in all 2-d arrays
   * @param[in]  rank - radius of the square region in which central pixel has a maximal value
   * @param[out] local_minima  - pointer to map of local minimums
   */
//-----------------------------

template <typename T>
size_t 
mapOfLocalMinimums(const T *data
                  ,const mask_t *mask
                  ,const size_t& rows
                  ,const size_t& cols
                  ,const size_t& rank
                  ,extrim_t *local_minima
                  )
{
  #ifndef NDEBUG
  std::cout << "XXX mapOfLocalMinimums point E \n";
  #endif

  // initialization of indexes
  //if(v_inddiag.empty())   
  std::vector<TwoIndexes> v_inddiag = evaluateDiagIndexes(rank);

  extrim_t *_local_minima = local_minima;
  size_t size = rows*cols;

  //if(_local_minima.empty()) 
  //   _local_minima = make_ndarray<extrim_t>(data.shape()[0], data.shape()[1]);
  std::fill_n(_local_minima, int(size), extrim_t(0));

  size_t counter = 0;

  unsigned rmin = 0;
  unsigned rmax = (int)rows;
  unsigned cmin = 0;
  unsigned cmax = (int)cols;
  int irank = (int)rank;

  int irc=0;
  int ircd=0;
  int irdc=0;
  int iric=0;

  // check rank minimum in columns and set the 1st bit (1)
  for(unsigned r = rmin; r<rmax; r++) {
    for(unsigned c = cmin; c<cmax; c++) {
      irc = r*cols+c;

      if(!mask[irc]) continue;
      if((c+1<cmax) && !(data[irc]<data[irc+1])) continue;
      _local_minima[irc] = 1;

      // positive side of c 
      unsigned dmax = min((int)cmax-1, int(c)+irank);
      for(unsigned cd=c+1; cd<=dmax; cd++) {
        ircd = r*cols+cd;
	if(mask[ircd] && (data[ircd] < data[irc])) { 
          _local_minima[irc] &=~1; // clear 1st bit
          c=cd-1; // jump ahead 
	  break;
	}
      }

      if(_local_minima[irc] & 1) {
        // negative side of c 
        unsigned dmin = max((int)cmin, int(c)-irank);
        for(unsigned cd=dmin; cd<c; cd++) {
          ircd = r*cols+cd;
	  if(mask[ircd] && (data[ircd] < data[irc])) { 
            _local_minima[irc] &=~1; // clear 1st bit
            c=cd+rank; // jump ahead 
	    break;
	  }
        }
      }

      // (r,c) is a local dip, jump ahead through the tested rank range
      if(!(c<cmax)) break;
      if(_local_minima[irc] & 1) c+=rank;
    }
  }

  // check rank minimum in rows and set the 2nd bit (2)
  for(unsigned c = cmin; c<cmax; c++) {
    for(unsigned r = rmin; r<rmax; r++) {
      // if it is not a local maximum from previous algorithm
      //if(!_local_minima[irc]) continue;
      irc = r*cols+c;

      if(!mask[irc]) continue;
      if((r+1<rmax) && !(data[irc]<data[irc+cols])) continue;
      _local_minima[irc] |= 2; // set 2nd bit

      // positive side of r 
      unsigned dmax = min((int)rmax-1, int(r)+irank);
      for(unsigned rd=r+1; rd<=dmax; rd++) {
        irdc = rd*cols+c;
	if(mask[irdc] && (data[irdc] < data[irc])) { 
          _local_minima[irc] &=~2; // clear 2nd bit
          r=rd-1; // jump ahead 
	  break;
	}
      }

      if(_local_minima[irc] & 2) {
        // negative side of r
        unsigned dmin = max((int)rmin, int(r)-irank);
        for(unsigned rd=dmin; rd<r; rd++) {
          irdc = rd*cols+c;
	  if(mask[irdc] && (data[irdc] < data[irc])) { 
            _local_minima[irc] &=~2; // clear 2nd bit
            r=rd+rank; // jump ahead 
	    break;
	  }
        }
      }

      // (r,c) is a local dip, jump ahead through the tested rank range
      if(!(r<rmax)) break;
      if(_local_minima[irc] & 2) r+=rank;
    }
  }

  // check rank minimum in "diagonal" regions and set the 3rd bit (4)
  for(unsigned r = rmin; r<rmax; r++) {
    for(unsigned c = cmin; c<cmax; c++) {
      irc = r*cols+c;
      // if it is not a local minimum from two previous algorithm
      if(_local_minima[irc] != 3) continue;
      _local_minima[irc] |= 4; // set 3rd bit

      for(vector<TwoIndexes>::const_iterator ij  = v_inddiag.begin();
                                             ij != v_inddiag.end(); ij++) {
        int ir = r + (ij->i);
        int ic = c + (ij->j);

        if(  ir<(int)rmin)  continue;
        if(  ic<(int)cmin)  continue;
        if(!(ir<(int)rmax)) continue;
        if(!(ic<(int)cmax)) continue;

        iric = ir*cols+ic;
	if(mask[iric] && (data[iric] < data[irc])) {
          _local_minima[irc] &=~4; // clear 3rd bit
	  break;
	}
      }

      // (r,c) is a local peak, jump ahead through the tested rank range
      if(_local_minima[irc] & 4) {
         c+=rank;
	 counter ++;
      }
    }
  }
  return counter;
}

//--------------------
  /**
   * @brief returns map of local maximums of requested rank, 
   *        where rank defins a square region around central pixel [rowc-rank, rowc+rank], [colc-rank, colc+rank].
   * 
   * Map of local maximum is a 2-d array of (uint16) values of data shape, 
   * with 0/+1/+2/+4 for non-maximum / maximum in column / maximum in row / minimum in square of radius rank.   
   * @param[in]  data - pointer to data array
   * @param[in]  mask - pointer to mask array; mask marks bad/good (0/1) pixels
   * @param[in]  rows - number of rows in all 2-d arrays
   * @param[in]  cols - number of columns in all 2-d arrays
   * @param[in]  rank - radius of the square region in which central pixel has a maximal value
   * @param[out] local_maxima  - pointer to map of local maximums
   */

template <typename T>
size_t 
mapOfLocalMaximums(const T *data
                  ,const mask_t *mask
                  ,const size_t& rows
                  ,const size_t& cols
                  ,const size_t& rank
                  ,extrim_t *local_maxima
                  )
{
  #ifndef NDEBUG
  std::cout << "in mapOfLocalMaximums, rank=" << rank << "\n";
  #endif

  // initialization of indexes
  std::vector<TwoIndexes> v_inddiag = evaluateDiagIndexes(rank);

  extrim_t *_local_maxima = local_maxima;
  std::fill_n(&_local_maxima[0], int(rows*cols), extrim_t(0));

  size_t counter = 0;

  unsigned rmin = 0;
  unsigned rmax = (int)rows;
  unsigned cmin = 0;
  unsigned cmax = (int)cols;
  int irank = (int)rank;

  int irc=0;
  int ircd=0;
  int irdc=0;
  int iric=0;

  // check rank maximum in columns and set the 1st bit (1)
  for(unsigned r = rmin; r<rmax; r++) {
    for(unsigned c = cmin; c<cmax; c++) {
      irc = r*cols+c;
      if(!mask[irc]) continue;
      if((c+1<cmax) && !(data[irc]>data[irc+1])) continue;
      _local_maxima[irc] = 1;

      // positive side of c 
      unsigned dmax = min((int)cmax-1, int(c)+irank);
      for(unsigned cd=c+1; cd<=dmax; cd++) {
        ircd = r*cols+cd;
	if(mask[ircd] && (data[ircd] > data[irc])) { 
          _local_maxima[irc] &=~1; // clear 1st bit
          c=cd-1; // jump ahead 
	  break;
	}
      }

      if(_local_maxima[irc] & 1) {
        // negative side of c 
        unsigned dmin = max((int)cmin, int(c)-irank);
        for(unsigned cd=dmin; cd<c; cd++) {
          ircd = r*cols+cd;
	  if(mask[ircd] && (data[ircd] > data[irc])) { 
            _local_maxima[irc] &=~1; // clear 1st bit
            c=cd+rank; // jump ahead 
	    break;
	  }
        }
      }

      // (r,c) is a local dip, jump ahead through the tested rank range
      if(!(c<cmax)) break;
      if(_local_maxima[irc] & 1) c+=rank;
    }
  }

  // check rank maximum in rows and set the 2nd bit (2)
  for(unsigned c = cmin; c<cmax; c++) {
    for(unsigned r = rmin; r<rmax; r++) {
      irc = r*cols+c;
      // if it is not a local maximum from previous algorithm
      //if(!_local_maxima[irc]) continue;

      if(!mask[irc]) continue;
      if((r+1<rmax) && !(data[irc]>data[irc+cols])) continue;
      _local_maxima[irc] |= 2; // set 2nd bit

      // positive side of r 
      unsigned dmax = min((int)rmax-1, int(r)+irank);
      for(unsigned rd=r+1; rd<=dmax; rd++) {
        irdc = rd*cols+c;
	if(mask[irdc] && (data[irdc] > data[irc])) { 
          _local_maxima[irc] &=~2; // clear 2nd bit
          r=rd-1; // jump ahead 
	  break;
	}
      }

      if(_local_maxima[irc] & 2) {
        // negative side of r
        unsigned dmin = max((int)rmin, int(r)-irank);
        for(unsigned rd=dmin; rd<r; rd++) {
          irdc = rd*cols+c;
	  if(mask[irdc] && (data[irdc] > data[irc])) { 
            _local_maxima[irc] &=~2; // clear 2nd bit
            r=rd+rank; // jump ahead 
	    break;
	  }
        }
      }

      // (r,c) is a local dip, jump ahead through the tested rank range
      if(!(r<rmax)) break;
      if(_local_maxima[irc] & 2) r+=rank;
    }
  }

  // check rank maximum in "diagonal" regions and set the 3rd bit (4)
  for(unsigned r = rmin; r<rmax; r++) {
    for(unsigned c = cmin; c<cmax; c++) {
      // if it is not a local maximum from two previous algorithm
      irc = r*cols+c;

      if(_local_maxima[irc] != 3) continue;
      _local_maxima[irc] |= 4; // set 3rd bit

      for(vector<TwoIndexes>::const_iterator ij  = v_inddiag.begin();
                                             ij != v_inddiag.end(); ij++) {
        int ir = r + (ij->i);
        int ic = c + (ij->j);

        if(  ir<(int)rmin)  continue;
        if(  ic<(int)cmin)  continue;
        if(!(ir<(int)rmax)) continue;
        if(!(ic<(int)cmax)) continue;

        iric = ir*cols+ic;
	if(mask[iric] && (data[iric] > data[irc])) {
          _local_maxima[irc] &=~4; // clear 3rd bit
	  break;
	}
      }

      // (r,c) is a local peak, jump ahead through the tested rank range
      if(_local_maxima[irc] & 4) {
         c+=rank;
	 counter ++;
      }
    }
  }
  return counter;
}

//-----------------------------
//-----------------------------
//-----------------------------
} // namespace localextrema
//-----------------------------
#endif // PSALGOS_LOCALEXTREMA_H
//-----------------------------
