#pragma once

#include <stdint.h>                     // For uint8_t

namespace SLEEK {

class SingleCompressorLossy
{
public:
  SingleCompressorLossy(size_t insize, float errorbound);
  ~SingleCompressorLossy();

  void banner() const;
  long long maxSize() const { return _maxsize; }
  void updateGraph(cudaStream_t         stream,
                   unsigned*      const state_d,
                   unsigned*      const index_d,
                   uint8_t const* const d_input_base,
                   long long      const inBufSize,
                   uint8_t*       const d_encoded_base,
                   long long      const encBufSize);
private:
  int _initialize(size_t inSize, float errorBound);
private:
  int        _eb_e;
  int        _thr_e;
  int        _offs;
  int        _blocks;
  long long  _chunks;
  long long* _d_fullcarry;
  long long  _maxsize;
};

} // SLEEK
