#pragma once

#include <stdint.h>                     // For uint8_t

namespace LC_framework {

class Compressor
{
  using byte = unsigned char;
public:
  Compressor(size_t insize, double paramv);
  ~Compressor();

  void banner() const;
  long long maxSize() const { return _maxsize; }
  void updateGraph(cudaStream_t         stream,
                   unsigned*      const state_d,
                   unsigned*      const index_d,
                   uint8_t const* const d_input_base,
                   long long      const insize,
                   uint8_t*       const d_encoded_base,
                   long long      const d_encsize);
private:
  double     _paramv[1];
  int        _blocks;
  byte*      _dpreencdata;
  long long* _d_fullcarry;
  long long  _maxsize;
};

} // LC_framework
