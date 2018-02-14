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
  EpicsCA (name,&_cb)
{
}

PvServer::~PvServer()
{
}

void PvServer::connected(bool v) 
{
  EpicsCA::connected(v);
  printf("PvServer[%s] connected %c\n",_channel.epicsName(),v?'T':'F');
  if (v)
    _channel.get();
}

void PvServer::updated() 
{
  printf("PvServer[%s] updated\n",_channel.epicsName());
}

#define handle_type(ctype, stype, dtype) case ctype:    \
  { dtype* inp  = (dtype*)data();                       \
    if (sz==2) {                                        \
      uint16_t* outp = (uint16_t*)payload;              \
      for(int k=0; k<nelem; k++) *outp++ = *inp++;      \
      result = (char*)outp - (char*)payload;            \
    } else {                                            \
      uint32_t* outp = (uint32_t*)payload;              \
      for(int k=0; k<nelem; k++) *outp++ = *inp++;      \
      result = (char*)outp - (char*)payload;            \
    }                                                   \
  }

/*
void PvServer::get() 
{
  _channel.get(); 
}
*/

int PvServer::fetch(char* payload, size_t sz)
{
  printf("PvServer[%s] fetch %p\n",_channel.epicsName(),payload);
  int result = 0;
  int nelem = _channel.nelements();
  switch(_channel.type()) {
    handle_type(DBR_TIME_SHORT , dbr_time_short , dbr_short_t ) break;
    handle_type(DBR_TIME_FLOAT , dbr_time_float , dbr_float_t ) break;
    handle_type(DBR_TIME_ENUM  , dbr_time_enum  , dbr_enum_t  ) break;
    handle_type(DBR_TIME_LONG  , dbr_time_long  , dbr_long_t  ) break;
    handle_type(DBR_TIME_DOUBLE, dbr_time_double, dbr_double_t) break;
    handle_type(DBR_TIME_CHAR  , dbr_time_char  , dbr_char_t  ) break;
  default: printf("Unknown type %d\n", int(_channel.type())); result=-1; break;
  }
  return result;
}

void PvServer::update() { _channel.get(); }
