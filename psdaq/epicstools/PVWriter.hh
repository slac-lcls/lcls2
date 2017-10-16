#ifndef Pds_PVWriter_hh
#define Pds_PVWriter_hh

#include "psdaq/epicstools/EpicsCA.hh"

namespace Pds_Epics {
  class PVWriter : public EpicsCA {
  public:
    PVWriter(const char* pvName, unsigned nelem) : EpicsCA(pvName,0,nelem) {}
    PVWriter(const char* pvName) : EpicsCA(pvName,0) {}
    ~PVWriter() {}
  public:
    void  put() { if (connected()) _channel.put(); }
  };

};

#endif
