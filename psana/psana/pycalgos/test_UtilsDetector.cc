
// g++ -O3 -o test_UtilsDetector -g test_UtilsDetector.cc
// ../lcls2/psana/psana/pycalgos/test_UtilsDetector
// mpirun -n  4 ../lcls2/psana/psana/pycalgos/test_UtilsDetector
// mpirun -n  64 ../lcls2/psana/psana/pycalgos/test_UtilsDetector

#include <stdint.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
//#include <cstdint>  // uint8_t

//#include "pycalgos/UtilsDetector.hh"
#include "UtilsDetector.hh"

using namespace utilsdetector;

//#define NGRINDS utilsdetector::NGRINDS // 4
//#define NPIXELS utilsdetector::NPIXELS // 16777216
//#define CCSV3   utilsdetector::CCSV3
//#define M14 0x3fff  // 16383 or (1<<14)-1 - 14-bit mask

// void calibrate(uint16_t* raw, uint8_t* mask, float* gain, float* ped, float* result) {
//  uint16_t* end = raw+SIZE;
//  while (raw<end) {
//    *result = ((*raw & M14) - *ped)*(*gain)*(*mask);
//    raw++; ped++; gain++; mask++; result++;
//  }
// }

void fill_CCSV3() {
  for (int n=0; n<NGRINDS; n++) {
    std::cout << "Gain bits combination: " << n << std::endl;
    for (int i=0; i<NPIXELS; i++) {
      CCSV3[n][i].pedestal = 1000 + n;
      CCSV3[n][i].gain = 10 + n;
      if (i<10) {
	std::cout << "pixel: " << i << " pedestal: " <<  CCSV3[n][i].pedestal << " gain: " << CCSV3[n][i].gain << std::endl;
      }
    }
  }
}

int main() {

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  fill_CCSV3();
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Time to fill CCSV3 = " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) << "[us]" << std::endl;

  //  uint16_t* raw = (uint16_t*)malloc(EVENTS*SIZE*sizeof(uint16_t));
  //  uint8_t* mask = (uint8_t*)malloc(SIZE*sizeof(uint8_t));
  //  float* result = (float*)malloc(SIZE*sizeof(float));
  //  float* ped = (float*)malloc(SIZE*sizeof(float));
  //  float* gain = (float*)malloc(SIZE*sizeof(float));

}
