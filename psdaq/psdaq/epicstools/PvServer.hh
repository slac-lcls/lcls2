/**
 * @file
 * @brief PvServer, an EpicsPVA whose updates go to a file-static callback that ignores them.
 */
#ifndef Pds_PvServer_hh
#define Pds_PvServer_hh

#include "psdaq/epicstools/EpicsPVA.hh"

namespace Pds_Epics {
  /** EpicsPVA whose monitor updates go to a file-static callback that ignores them, so its own updated() is not called by EpicsPVA. Only named in a using-declaration in hsd/src/PVCtrlsBase.cc. */
  class PvServer : public EpicsPVA,
                   public PVMonitorCb {
  public:
    /** Connect to the PV through pva with the file-static callback. */
    PvServer(const char*);
    /** Does nothing; empty body. */
    ~PvServer();
  public:
    /** Declared but not defined in PvServer.cc. */
    void connected(bool);
    /** Does nothing; empty body. */
    void updated();
  public:
    /** Call get() on the channel and discard the result. */
    void update     ();
  };
};

#endif
