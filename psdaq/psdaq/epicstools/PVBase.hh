#ifndef Pds_PVBase_hh
#define Pds_PVBase_hh

#include "psdaq/epicstools/EpicsPVA.hh"

//
//  Abstract base class for read/write channel access
//
namespace Pds_Epics {
  class PVBase : public EpicsPVA,
                 public PVMonitorCb {
  public:
    PVBase(const char* channelName, const int maxElements=1) : EpicsPVA(channelName, this, maxElements) {}
    ~PVBase() {}
    void updated() {}
  };
};

#endif
