#ifndef Pds_PVBase_hh
#define Pds_PVBase_hh

#include "psdaq/epicstools/EpicsCA.hh"

//
//  Abstract base class for read/write channel access
//
namespace Pds_Epics {
  class PVBase : public EpicsCA,
                 public PVMonitorCb {
  public:
    PVBase(const char* channelName, const int maxElements=1) : EpicsCA(channelName, this, maxElements) {}
    virtual ~PVBase() {}
  };
};

#endif
