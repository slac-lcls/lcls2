#include "psdaq/epicstools/PvServer.hh"
#include <stdio.h>
#include <stdint.h>

using namespace Pds_Epics;

class PvServerCb : public PVMonitorCb {
public:
  PvServerCb() {}
  void updated() {}
};

static PvServerCb _cb;

PvServer::PvServer(const char* name) :
  EpicsPVA (name,&_cb)
{
}

PvServer::~PvServer()
{
}


void PvServer::update() { _channel.get(); }

void PvServer::updated() { }
