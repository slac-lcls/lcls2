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
  FileWriter(size_t bufferSize, bool dio);
  ~FileWriter() override;
  static void dumpProperties();
  int registerStream(cudaStream_t);
  int open(const std::string& fileName) override;
  int close() override;
  void writeEvent(const void* devPtr, size_t size, const XtcData::TimeStamp) override;
protected:
  virtual ssize_t _write();
protected:
  cudaStream_t       m_stream;
  int                m_fd;
  CUfileHandle_t     m_handle;
  size_t             m_count;
  XtcData::TimeStamp m_batch_starttime;
  size_t             m_bufferSize;
  uint8_t*           m_buffer_d;        // device pointer
  off_t              m_fileOffset;
  bool               m_dio;
};

class FileWriterAsync : public FileWriter
{
public:
  FileWriterAsync(size_t bufferSize, bool dio);
private:
  ssize_t _write() override;
};

  } // Gpu
} // Drp
