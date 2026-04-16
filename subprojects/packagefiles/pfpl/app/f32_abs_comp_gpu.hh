#pragma once

namespace PFPL {

class PFPL_Compressor
{
  using byte = unsigned char;
public:
  PFPL_Compressor(size_t insize, float errorbound);
  PFPL_Compressor(size_t insize, float errorbound, float threshold);
  ~PFPL_Compressor();

  void banner() const;
  long long maxSize() const { return _maxsize; }
  void updateGraph(cudaStream_t      stream,
                   const unsigned&   index,
                   byte const* const d_input_base,
                   const long long   inBufSize,
                   byte* const       d_encoded_base,
                   const long long   encBufSize);
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
