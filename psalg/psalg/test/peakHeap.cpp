//g++ -Wall -std=c++11 -I /reg/neh/home/yoon82/temp/lcls2/install/include peakHeap.cpp psalg/src/PeakFinderAlgos.cpp psalg/src/LocalExtrema.cpp -o peakHeap
// To turn off debug:
//g++ -Wall -std=c++11 -I /reg/neh/home/yoon82/temp/lcls2/install/include peakHeap.cpp psalg/src/PeakFinderAlgos.cpp psalg/src/LocalExtrema.cpp -DNDEBUG -o peakHeap

#include <iostream>
#include <stdlib.h>

#include "psalg/include/PeakFinderAlgos.h"
#include "psalg/include/LocalExtrema.h"
#include "psalg/include/Types.h"

//#include "xtcdata/xtc/DescData.hh" // Array
#include "psalg/include/Array.hh"
#include "psalg/include/Allocator.hh"

#include <chrono> // timer
typedef std::chrono::high_resolution_clock Clock;

using namespace psalgos;
using namespace temp;

int main () {

  //Heap *hptr = new Heap;
  //Stack *sptr = new Stack;
  //Stack *sptr1 = new Stack;
  Stack stack;
  Stack stack1;

  // Step 0: fake data and mask
  unsigned int rows = 185;
  unsigned int cols = 388;
  int16_t *data  = new int16_t[rows*cols];
  int16_t *data1 = new int16_t[rows*cols];
  for(unsigned int i=0; i<rows*cols; i++) {
      data[i]  = rand() % 10;
      data1[i] = data[i];
  }
  data[1900] = 1000; // peak 1
  data[1901] = 900;
  data[1902] = 900;
  data[5900] = 500; // peak 2
  data[5901] = 800;
  data[5902] = 300;

  data1[900] = 1000; // peak 1
  data1[901] = 900;
  data1[902] = 900;
  data1[3900] = 500; // peak 2
  data1[3901] = 800;
  data1[3902] = 300;
  uint16_t *mask = new uint16_t[rows*cols];
  for(unsigned int i=0; i<rows*cols; i++) {
      mask[i] = 1;
  }

  // Step 1: Init PeakFinderAlgos
  const size_t  seg = 0;
  const unsigned pbits = 0;

  PeakFinderAlgos *ptr;
  ptr = new PeakFinderAlgos(&stack, seg, pbits);

  // Step 2: Set params
  const float npix_min = 2;
  const float npix_max = 30;
  const float amax_thr = 200;
  const float atot_thr = 600;
  const float son_min = 7;
  ptr->setPeakSelectionPars(npix_min, npix_max, amax_thr, atot_thr, son_min);

  // Step 3: Peak finder
  const size_t rank = 3;
  const double r0 = 4;
  const double dr = 2;
  const double nsigm = 0;

  auto tic = Clock::now();
  ptr->peakFinderV3r3(data, mask, rows, cols, rank, r0, dr, nsigm);
  auto toc = Clock::now();
  ptr->_printVectorOfPeaks_drp(ptr->vectorOfPeaksSelected_drp());
  std::cout << chrono::duration_cast<chrono::microseconds>(toc - tic).count() << " microseconds" << std::endl;

  ptr->setAllocator(&stack1); // simulate PEBBLE

  tic = Clock::now();
  ptr->peakFinderV3r3(data1, mask, rows, cols, rank, r0, dr, nsigm);
  toc = Clock::now();
  ptr->_printVectorOfPeaks_drp(ptr->vectorOfPeaksSelected_drp());

  std::cout << chrono::duration_cast<chrono::microseconds>(toc - tic).count() << " microseconds" << std::endl;

  tic = Clock::now();
  int numEvents = 10;
  for(int i = 0; i < numEvents; i++){
    if(i%2==0) {
      ptr->peakFinderV3r3(data, mask, rows, cols, rank, r0, dr, nsigm);
    } else {
      ptr->peakFinderV3r3(data1, mask, rows, cols, rank, r0, dr, nsigm);
    }
    //ptr->_printVectorOfPeaks_drp(ptr->vectorOfPeaksSelected_drp());
  }
  toc = Clock::now();
  std::cout << "Delta t: " << std::endl
            << std::chrono::duration_cast<std::chrono::microseconds>(toc - tic).count() / numEvents
            << " microseconds" << std::endl;

// DATA
// Peak 1
//Seg:  0 Row:   4 Col: 348 Npix: 24 Imax:  995.7 Itot: 2843.5 CGrav r:   4.0 c: 349.0 Sigma r: 0.29 c: 0.86 Rows[   1:   7] Cols[ 345: 351] B:  4.3 N:  1.8 S/N:314.7
// Peak 2
//Seg:  0 Row:  15 Col:  81 Npix: 23 Imax:  796.0 Itot: 1637.3 CGrav r:  15.0 c:  80.8 Sigma r: 0.35 c: 0.74 Rows[  12:  18] Cols[  78:  83] B:  4.0 N:  2.2 S/N:157.5

// DATA1
// Peak 1
//Seg:  0 Row:   2 Col: 124 Npix: 18 Imax:  994.6 Itot: 2808.5 CGrav r:   2.0 c: 125.0 Sigma r: 0.21 c: 0.85 Rows[   1:   5] Cols[ 121: 127] B:  5.4 N:  2.3 S/N:289.4
// Peak 2
//Seg:  0 Row:  10 Col:  21 Npix: 16 Imax:  795.4 Itot: 1611.8 CGrav r:  10.0 c:  20.9 Sigma r: 0.28 c: 0.71 Rows[   7:  13] Cols[  18:  22] B:  4.6 N:  1.7 S/N:241.4

  delete[] data;
  delete[] data1;
  delete[] mask;
  delete ptr;

  return 0;
}
