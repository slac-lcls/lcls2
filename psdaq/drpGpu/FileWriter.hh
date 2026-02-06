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
private:
  virtual void _reset();
  virtual void _flush();
  ssize_t _write();
protected:
  cudaStream_t       m_stream;
  CUfileHandle_t     m_handle;
  XtcData::TimeStamp m_batch_starttime;
  uint8_t*           m_buffer_d;        // device pointer
  off_t              m_fileOffset;
  size_t             m_bufferSize;
  int                m_fd;
  bool               m_dio;
private:
  size_t             m_count;
};

class FileWriterAsync : public FileWriter
{
public:
  FileWriterAsync(size_t bufferSize, bool dio);
  void writeEvent(const void* devPtr, size_t size, const XtcData::TimeStamp) override;
protected:
  void _reset() override;
  void _flush() override;
private:
  void _write();
private:
  size_t   m_counts[2];
  off_t    m_bufOffset[2];
  size_t   m_index;
  ssize_t  m_bytesWritten;
};

  } // Gpu
} // Drp
