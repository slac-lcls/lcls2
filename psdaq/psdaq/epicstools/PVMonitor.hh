#ifndef Pds_PVMonitor_hh
#define Pds_PVMonitor_hh

#include "psdaq/epicstools/EpicsPVA.hh"

namespace Pds_Epics {
  class PVMonitor : public EpicsPVA {
  public:
    PVMonitor(const char* pvName, PVMonitorCb& cb) : EpicsPVA(pvName,&cb) {}
    ~PVMonitor() {}
  };

};

#endif
