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
  virtual Detector* detector() = 0;
  virtual void timingHeaders(unsigned index, Pds::TimingHeader* buffer) = 0;
  virtual void process(Batch& batch, bool& sawDisable) = 0;
  virtual void reader(uint32_t start, SPSCQueue<Batch>& collectorGpuQueue) = 0;
  virtual unsigned lastEvtCtr() const = 0;
};

};
