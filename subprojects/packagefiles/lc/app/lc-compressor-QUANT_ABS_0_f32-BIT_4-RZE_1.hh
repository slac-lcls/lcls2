#pragma once

namespace LC_framework {

class LC_Compressor
{
  using byte = unsigned char;
public:
  LC_Compressor(size_t insize, double paramv);
  ~LC_Compressor();

  void banner() const;
  long long maxSize() const { return _maxsize; }
  void updateGraph(cudaStream_t      stream,
                   const unsigned&   index,
                   byte const* const d_input_base,
                   const long long   insize,
                   byte* const       d_encoded_base,
                   const long long   d_encsize);
private:
  double     _paramv[1];
  int        _blocks;
  byte*      _dpreencdata;
  long long* _d_fullcarry;
  long long  _maxsize;
};

} // LC_framework
