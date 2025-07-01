#pragma once

#include <cstddef>                      // size_t
#include <cstdint>                      // uint*_t
#include <string>

#include <cuda_runtime.h>
#include "cufile.h"                     // CUfileHandle_t

#include "drp/FileWriterBase.hh"

namespace Drp {
  namespace Gpu {

class FileWriter : public Drp::FileWriterBase
{
public:
  FileWriter(size_t bufferSize, bool dio, cudaStream_t& stream);
  ~FileWriter() override;
  void dumpProperties() const;
  int open(const std::string& fileName) override;
  int close() override;
  void writeEvent(const void* devPtr, size_t size, const XtcData::TimeStamp ts) override;
private:
  ssize_t _write(const void* data, size_t size);
private:
  int                m_fd;
  cudaStream_t&      m_stream;
  CUfileHandle_t     m_handle;
  size_t             m_count;
  XtcData::TimeStamp m_batch_starttime;
  size_t             m_bufferSize;
  uint8_t*           m_buffer_d;        // device pointer
  off_t              m_fileOffset;
  bool               m_dio;
};

  } // Gpu
} // Drp
