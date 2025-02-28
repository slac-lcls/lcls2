
// g++ -O3 -o test_cpo -g test_cpo.cc
// ../lcls2/psana/psana/pycalgos/test_cpo
// mpirun -n  4 ../lcls2/psana/psana/pycalgos/test_cpo
// mpirun -n  64 ../lcls2/psana/psana/pycalgos/test_cpo

#define NLOOPS 100
#define EVENTS 100
#define SIZE 16*352*384
#define M14 0x3fff  // 16383 or (1<<14)-1 - 14-bit mask

#include <stdint.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include <cstdint>  // uint8_t

void calibrate(uint16_t* raw, uint8_t* mask, float* gain, float* ped, float* result) {
  uint16_t* end = raw+SIZE;
  while (raw<end) {
    *result = ((*raw & M14) - *ped)*(*gain)*(*mask);
    raw++; ped++; gain++; mask++; result++;
  }
}

int main() {

  uint16_t* raw = (uint16_t*)malloc(EVENTS*SIZE*sizeof(uint16_t));
  uint8_t* mask = (uint8_t*)malloc(SIZE*sizeof(uint8_t));
  float* result = (float*)malloc(SIZE*sizeof(float));
  float* ped = (float*)malloc(SIZE*sizeof(float));
  float* gain = (float*)malloc(SIZE*sizeof(float));

  for (int i=0; i<EVENTS*SIZE; i++) {
    raw[i]=1234;
  }

  for (int i=0; i<SIZE; i++) {
    mask[i]=1;
    //result[i]=0.0;
    ped[i]=1233.1;
    gain[i]=1.234;
  }

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  for (int n=0; n<NLOOPS; n++){
    for (int i=0; i<EVENTS; i++){
      calibrate(raw+i*SIZE, mask, gain, ped, result);
    }
  }
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  std::cout << "NLOOPS: " << NLOOPS << " EVENTS: " << EVENTS << std::endl;
  std::cout << "Time per event = " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count())/EVENTS/NLOOPS << "[us]" << std::endl;
}
