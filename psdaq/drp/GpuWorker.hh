#pragma once

#include <cstddef>

namespace Pds {
  class TimingHeader;
};

template <typename T> class SPSCQueue;

namespace Drp {

class Detector;
struct Batch;

// Interface class returned from a shareable library
class GpuWorker
{
public:
  virtual ~GpuWorker() = default;
  virtual void reader(uint32_t start, SPSCQueue<Batch>& collectorGpuQueue) = 0;
  virtual unsigned lastEvtCtr() const = 0;
public:
  enum DmaMode_t { CPU=0x0000ffff, GPU=0xffff0000, ERR=-1u };
  virtual DmaMode_t dmaMode() const = 0;
  virtual void dmaMode(DmaMode_t mode_) = 0;
};

};