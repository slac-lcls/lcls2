#ifndef Pds_PvServer_hh
#define Pds_PvServer_hh

#include "psdaq/epicstools/EpicsPVA.hh"

namespace Pds_Epics {
  class PvServer : public EpicsPVA,
                   public PVMonitorCb {
  public:
    PvServer(const char*);
    ~PvServer();
  public:
    void connected(bool);
    void updated();
  public:
    void update     ();
  };
};

#endif
