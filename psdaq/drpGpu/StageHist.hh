#pragma once

#include <cstdint>
#include <string>

namespace Drp {
  namespace Gpu {

// Binned by floor(log2(ns)) of %globaltimer
static constexpr unsigned NLog2Bins{40};

// Always-on per-event service time of a graph stage, for the Prometheus metrics
struct StageTime
{
  uint64_t tStart;                      // When the current event's index was obtained
  uint64_t sum;                         // Summed service time, ns; adjacent to count
  uint64_t count;                       //   so the host reads both in one copy
};

// Per-event timing of a TrgInpGen or Reducer graph, kept on the device and logged at
// Unconfigure (-k readerHist=1)
struct StageHist
{
  uint64_t tStart;                      // When the current event's index was obtained
  uint64_t tWork;                       // When the loop kernel first saw the work done
  uint64_t tPrevStart;
  uint64_t interval[NLog2Bins];         // Between successive indices obtained
  uint64_t service [NLog2Bins];         // Index obtained to index pushed downstream
  uint64_t work    [NLog2Bins];         //   index obtained to work seen done, launches included
  uint64_t push    [NLog2Bins];         //   that to the index pushed
  uint64_t sum[4];                      // In the order above
  uint64_t pushFulls;                   // Push attempts abandoned on a full queue
};

#ifdef __CUDACC__
static __device__ __forceinline__ uint64_t globalTimer()
{
  uint64_t ns;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(ns));
  return ns;
}

static __device__ __forceinline__ void histLog2(uint64_t* const hist, uint64_t& sum, uint64_t const ns)
{
  auto bin{ns ? 63u - __clzll(ns) : 0u};
  if (bin >= NLog2Bins)  bin = NLog2Bins - 1;
  ++hist[bin];
  sum += ns;
}

static __device__ __forceinline__ void stageTimeStart(StageTime* const time)
{
  time->tStart = globalTimer();
}

static __device__ __forceinline__ void stageTimePushed(StageTime* const time)
{
  time->sum += globalTimer() - time->tStart;
  ++time->count;
}

static __device__ __forceinline__ void stageHistStart(StageHist* const hist)
{
  auto const t{globalTimer()};
  if (hist->tPrevStart)  histLog2(hist->interval, hist->sum[0], t - hist->tPrevStart);
  hist->tPrevStart = t;
  hist->tStart     = t;
}

// Called on every loop pass that finds the work done, until the push succeeds
static __device__ __forceinline__ void stageHistWorkDone(StageHist* const hist)
{
  if (!hist->tWork) {
    hist->tWork = globalTimer();
    histLog2(hist->work, hist->sum[2], hist->tWork - hist->tStart);
  }
}

static __device__ __forceinline__ void stageHistPushed(StageHist* const hist)
{
  auto const t{globalTimer()};
  histLog2(hist->service, hist->sum[1], t - hist->tStart);
  histLog2(hist->push,    hist->sum[3], t - hist->tWork);
  hist->tWork = 0;
}
#endif

// Mean service time in us of the events since the previous call, summed over n stages;
// false when there were none.  prev holds {sum, count} between calls.
bool stageTimeMean(StageTime* const* time_d, unsigned n, uint64_t prev[2], double& us);

std::string log2Line(const uint64_t* hist, unsigned nBins);

// Copies hist_d to the host and logs it; a snapshot, since the graph may still be running
void reportStageHist(const char* name, const StageHist* hist_d);

  } // Gpu
} // Drp
