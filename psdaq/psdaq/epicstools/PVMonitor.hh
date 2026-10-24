/**
 * @file
 * @brief PVMonitor, an EpicsPVA that reports updates to a separate PVMonitorCb.
 */
#ifndef Pds_PVMonitor_hh
#define Pds_PVMonitor_hh

#include "psdaq/epicstools/EpicsPVA.hh"

namespace Pds_Epics {
  /** EpicsPVA that reports updates to a separate PVMonitorCb; not used elsewhere in psdaq. */
  class PVMonitor : public EpicsPVA {
  public:
    /** Connect to pvName through pva with cb as the monitor callback. */
    PVMonitor(const char* pvName, PVMonitorCb& cb) : EpicsPVA(pvName,&cb) {}
    /** Does nothing; empty body. */
    ~PVMonitor() {}
  };

};

#endif
