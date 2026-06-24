#pragma once

#include <stdint.h>                     // For uint8_t

namespace PFPL {

class Compressor
{
  using byte = unsigned char;
public:
  Compressor(size_t insize, float errorbound);
  Compressor(size_t insize, float errorbound, float threshold);
  ~Compressor();

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
  void _initialize(size_t insize);
private:
  float _errorbound;
  float _threshold;
  int   _blocks;
  int*  _d_fullcarry;
  int   _maxsize;
};

} // PFPL
