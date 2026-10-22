#include "StageHist.hh"

#include "gpuUtils.hh"                  // For chkError
#include "psalg/utils/SysLog.hh"

using logging = psalg::SysLog;

namespace Drp {
  namespace Gpu {

bool stageTimeMean(StageTime* const* time_d, unsigned n, uint64_t prev[2], double& us)
{
  uint64_t sum{0}, count{0};
  for (unsigned i = 0; i < n; ++i) {
    uint64_t sumCount[2];
    if (chkError(cudaMemcpy(sumCount, &time_d[i]->sum, sizeof(sumCount), cudaMemcpyDeviceToHost)))  return false;
    sum += sumCount[0];  count += sumCount[1];
  }
  auto const dSum{sum - prev[0]}, dCount{count - prev[1]};
  prev[0] = sum;  prev[1] = count;
  if (!dCount)  return false;
  us = 1e-3 * double(dSum) / double(dCount);
  return true;
}

std::string log2Line(const uint64_t* hist, unsigned nBins)
{
  unsigned lo{nBins}, hi{0};
  for (unsigned i = 0; i < nBins; ++i) {
    if (hist[i]) { if (i < lo) lo = i;  hi = i; }
  }
  std::string line;
  for (unsigned i = lo; i <= hi && lo < nBins; ++i)
    line += " 2^" + std::to_string(i) + ":" + std::to_string(hist[i]);
  return line;
}

void reportStageHist(const char* name, const StageHist* hist_d)
{
  if (!hist_d)  return;
  StageHist sh;
  if (chkError(cudaMemcpy(&sh, hist_d, sizeof(sh), cudaMemcpyDeviceToHost)))  return;

  const char* field[] {"interval", "service", " work", " push"};
  const uint64_t* hist[] {sh.interval, sh.service, sh.work, sh.push};
  for (unsigned j = 0; j < 4; ++j) {
    uint64_t count{0};
    for (unsigned k = 0; k < NLog2Bins; ++k)  count += hist[j][k];
    logging::info("%s %-8s ns: n %lu, mean %.0f |%s", name, field[j], count,
                  count ? double(sh.sum[j]) / count : 0., log2Line(hist[j], NLog2Bins).c_str());
  }
  logging::info("%s failed attempts: push %lu", name, sh.pushFulls);
}

  } // Gpu
} // Drp
