#pragma once

#include "psdaq/service/dl.hh"

#include <cuda_runtime.h>


namespace Drp {
  namespace Gpu {

class ReducerAlgo;

class Reducer
{
public:
  Reducer(const Parameters&, MemPoolGpu&);
  ~Reducer(); // = default;
private:
  int _setupReducer();
  int _setupGraph();
private:
  MemPoolGpu&       m_pool;
  cudaStream_t      m_stream;
  Dl                m_dl;
  ReducerAlgo*      m_reducer;
  const Parameters& m_para;
};

  } // Gpu
} // Drp
