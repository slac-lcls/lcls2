//g++ -std=c++11 -I /reg/neh/home/yoon82/temp/lcls2 peak.cpp psalgos/src/PeakFinderAlgos.cpp psalgos/src/LocalExtrema.cpp -o peak

#include <iostream>
#include "psalgos/include/PeakFinderAlgos.h"
#include <vector>
#include <stdlib.h>

#include "psalgos/include/LocalExtrema.h"

//#include "xtcdata/xtc/DescData.hh" // struct Array, Vector
#include "psalgos/include/Types.h"

#include <chrono> // timer
typedef std::chrono::high_resolution_clock Clock;

using namespace psalgos;

int main () {

  // DRP: you are given PEBBLE, data, mask

  // Step 0: fake data and mask
  unsigned int rows = 185;
  unsigned int cols = 388;
  int16_t *data = new int16_t[rows*cols];
  for(unsigned int i=0; i<rows*cols; i++) {
      data[i] = rand() % 10;
  }
  data[1900] = 1000; // peak 1
  data[1901] = 900;
  data[1902] = 900;
  data[5900] = 500; // peak 2
  data[5901] = 800;
  data[5902] = 300;

  uint16_t *mask = new uint16_t[rows*cols];
  for(unsigned int i=0; i<rows*cols; i++) {
      mask[i] = 1;
  }

  //uint8_t *buf = NULL;
  uint8_t *buf = new uint8_t[10240*100000]; // PEBBLE fex_stack

  auto t1 = Clock::now();
    
  // Step 1: Init PeakFinderAlgos
  const size_t  seg = 0;
  const unsigned pbits = 1;
  if (pbits) std::cout << "+buf Address stored " << (void *) buf << std::endl;
  
  PeakFinderAlgos *ptr;
  if (!buf) {
      ptr = new PeakFinderAlgos(seg, pbits);
  } else {
      ptr = new(buf) PeakFinderAlgos(seg, pbits, buf+sizeof(PeakFinderAlgos)); // placement new
  }

  auto t2 = Clock::now();

  // Step 2: Set params
  const float npix_min = 2;
  const float npix_max = 30; 
  const float amax_thr = 200;
  const float atot_thr = 600;
  const float son_min = 7;
  ptr->setPeakSelectionPars(npix_min, npix_max, amax_thr, atot_thr, son_min);
  
  auto t3 = Clock::now();

  // Step 3: Peak finder
  const size_t rank = 3;
  const double r0 = 4;
  const double dr = 2;
  const double nsigm = 0;
  ptr->peakFinderV3r3(data, mask, rows, cols, rank, r0, dr, nsigm);


  auto t4 = Clock::now();
  std::cout << "Analyzing fake data, Delta t: " << std::endl 
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << std::endl
            << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count() << std::endl
            << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count() << std::endl
            << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t1).count() 
            << " milliseconds" << std::endl;


  std::cout << "*** BUF *** " << (void*) buf << std::endl; // TODO: update buf to the latest address

// Peak 1
//Seg:  0 Row:   4 Col: 348 Npix: 24 Imax:  995.7 Itot: 2843.5 CGrav r:   4.0 c: 349.0 Sigma r: 0.29 c: 0.86 Rows[   1:   7] Cols[ 345: 351] B:  4.3 N:  1.8 S/N:314.7
// Peak 2
//Seg:  0 Row:  15 Col:  81 Npix: 23 Imax:  796.0 Itot: 1637.3 CGrav r:  15.0 c:  80.8 Sigma r: 0.35 c: 0.74 Rows[  12:  18] Cols[  78:  83] B:  4.0 N:  2.2 S/N:157.5

  return 0;
}
