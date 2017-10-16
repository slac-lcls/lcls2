#ifndef Pds_PvServer_hh
#define Pds_PvServer_hh

#include "psdaq/epicstools/EpicsCA.hh"

namespace Pds_Epics {
  class PvServer : public EpicsCA,
                   public PVMonitorCb {
  public:
    PvServer(const char*);
    ~PvServer();
  public:
    void connected(bool);
    void updated();
  public:
    int  fetch      (char* copyTo, size_t wordSize);
    int  fetch_u32  (char* copyTo, size_t nwords, size_t offset=0);
    void update     ();
  };
};

#endif
