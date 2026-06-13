#include "psdaq/hsd/src/FexCfg.hh"
#include <unistd.h>

using namespace Pds::HSD;

void FexCfg::disable()
{
  _streams = 0;
}

void FexCfg::TriggerMonitor::resetCounters()
{
    unsigned v = apply_correction;
    v |= 1;
    apply_correction = v;
    usleep(10);
    v &= ~1;
    apply_correction = v;
}

void FexCfg::TriggerMonitor::enableCorrection(bool enable)
{
    unsigned v = apply_correction;
    if (enable)
        v |= 2;
    else
        v &= ~2;
    apply_correction = v;
}
