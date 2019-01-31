//--------------------------------------------------------------------------
//
// File and Version Information:
// 	$Id: EpicsCA.cc 6181 2016-07-07 01:06:10Z tookey@SLAC.STANFORD.EDU $
//
// Environment:
//      This software was developed for the BaBar collaboration.  If you
//      use all or part of it, please give an appropriate acknowledgement.
//
//
//------------------------------------------------------------------------

#include "psdaq/epicstools/EpicsCA.hh"

#include <string.h>
#include <stdlib.h>

// epics includes
#include "db_access.h"

#include <stdio.h>

//#define DBUG

static void ConnStatusCB(struct connection_handler_args chArgs)
{
  Pds_Epics::EpicsCAChannel* theProxy = (Pds_Epics::EpicsCAChannel*) ca_puser(chArgs.chid);
  theProxy->connStatusCallback(chArgs);
}
  
static void GetDataCallback(struct event_handler_args ehArgs)
{
  Pds_Epics::EpicsCAChannel* theProxy = (Pds_Epics::EpicsCAChannel*) ehArgs.usr;
  theProxy->getDataCallback(ehArgs);
}
  
static void PutDataCallback(struct event_handler_args ehArgs)
{
  Pds_Epics::EpicsCAChannel* theProxy = (Pds_Epics::EpicsCAChannel*) ehArgs.usr;
  theProxy->putDataCallback(ehArgs);
}


#define handle_type(ctype, stype, dtype) case ctype: \
  { struct stype* ival = (struct stype*)dbr;         \
    _stamp      = ival->stamp;                       \
    _status     = ival->status;                      \
    _severity   = ival->severity;                    \
    dtype* inp  = &ival->value;                      \
    dtype* outp = (dtype*)_pvdata;                   \
    for(int k=0; k<nelem; k++) *outp++ = *inp++;     \
  }

using namespace Pds_Epics;

EpicsCA::EpicsCA(const char*   channelName,
                 PVMonitorCb*  monitor,
                 const int maxElements) :
  _channel(channelName,monitor!=0,*this,maxElements),
  _monitor(monitor),
  _pvdata(new char[1]),
  _pvsiz(0),
  _connected(false)
{
}

EpicsCA::~EpicsCA()
{
  delete[] _pvdata; 
}

void EpicsCA::connected   (bool c) 
{
  int nelem = _channel.nelements();
  int sz = dbr_size_n(_channel.type(),nelem);
  if (_pvsiz < sz) {
    delete[] _pvdata;
    _pvdata = new char[_pvsiz=sz];
#ifdef DBUG
    printf("pvdata allocated @ %p sz %d type %d\n",_pvdata,sz,int(_channel.type()));
#endif
  }
  else {
#ifdef DBUG
    printf("pvdata retained @ %p sz %d type %d\n",_pvdata,sz,int(_channel.type()));
#endif
  }
  _connected = c;
}

bool EpicsCA::connected() const { return _connected; }

void EpicsCA::getData     (const void* dbr)  
{
  int nelem = _channel.nelements();
  switch(_channel.type()) {
    handle_type(DBR_TIME_SHORT , dbr_time_short , dbr_short_t ) break;
    handle_type(DBR_TIME_FLOAT , dbr_time_float , dbr_float_t ) break;
    handle_type(DBR_TIME_ENUM  , dbr_time_enum  , dbr_enum_t  ) break;
    handle_type(DBR_TIME_LONG  , dbr_time_long  , dbr_long_t  ) break;
    handle_type(DBR_TIME_DOUBLE, dbr_time_double, dbr_double_t) break;
    handle_type(DBR_TIME_CHAR  , dbr_time_char  , dbr_char_t  ) break;
  default: printf("Unknown type %d\n", int(_channel.type())); break;
  }

  if (_monitor) _monitor->updated();
}

void* EpicsCA::data        () 
{
  return _pvdata; 
}

unsigned EpicsCA::sec () const { return _stamp.secPastEpoch; }
unsigned EpicsCA::nsec() const { return _stamp.nsec; }

size_t EpicsCA::data_size  () const { return _pvsiz; }

void  EpicsCA::putStatus   (bool s) {}


EpicsCAChannel::EpicsCAChannel(const char* channelName,
                               bool        monitor,
                               EpicsCA&    proxy,
                               const int   maxElements) :
  _maxElements(maxElements),
  _connected  (NotConnected),
  _monitor    (monitor),
  _monitored  (false),
  _proxy      (proxy)
{
  snprintf(_epicsName, 64, channelName);
  strtok(_epicsName, "[");
//   char* index = strtok(NULL,"]");
//   if (index)
//     sscanf(index,"%d",&_element);
//   else
//     _element=0;

  const int priority = 0;
  int st = ca_create_channel(_epicsName, ConnStatusCB, this, priority, &_epicsChanID);
  if (st != ECA_NORMAL) 
    printf("EpicsCAChannel::ctor %s : %s\n", _epicsName, ca_message(st));

#ifdef DBUG
  printf("EpicsCAChannel ca_create_channel[%s]\n",_epicsName);
#endif  
}

EpicsCAChannel::~EpicsCAChannel()
{
  if (_connected != NotConnected && _monitor)
    ca_clear_subscription(_event);

  ca_clear_channel(_epicsChanID);
}

void EpicsCAChannel::get()
{
#ifdef DBUG
  printf("EpicsCAChannel::get[%s]\n",_epicsName);
#endif  

  int st = ca_array_get_callback (_type,
				  _nelements,
				  _epicsChanID,
				  GetDataCallback,
				  this);
  if (st != ECA_NORMAL)
    printf("%s : %s [get st]\n",_epicsName, ca_message(st));
}

void EpicsCAChannel::put()
{
  int dbfType = ca_field_type(_epicsChanID);
  int st = ca_array_put (dbfType, 
			 _nelements,
			 _epicsChanID,
			 _proxy.data());
  if (st != ECA_NORMAL)
    printf("%s : %s [put st] : %d\n",_epicsName, ca_message(st), dbfType);
}

void EpicsCAChannel::put_cb()
{
  int dbfType = ca_field_type(_epicsChanID);
  int st = ca_array_put_callback (dbfType,
				  _nelements,
				  _epicsChanID,
				  _proxy.data(),
				  PutDataCallback,
				  this);
  if (st != ECA_NORMAL)
    printf("%s : %s [put_cb st] : %d\n",_epicsName, ca_message(st), dbfType);
}

void EpicsCAChannel::connStatusCallback(struct connection_handler_args chArgs)
{
#ifdef DBUG
  printf("EpicsCAChannel::connStatusCallback[%s]\n",_epicsName);
#endif  

  if ( chArgs.op == CA_OP_CONN_UP ) {
    _connected = Connected;
    int dbfType = ca_field_type(_epicsChanID);

    int dbrType = dbf_type_to_DBR_TIME(dbfType);
    if (dbr_type_is_ENUM(dbrType))
      dbrType = DBR_TIME_INT;
    
    _type = dbrType;
    _nelements = ca_element_count(_epicsChanID);
    if (_maxElements > 0 && _nelements > _maxElements) {
#ifdef DBUG
      printf("EpicsCAChannel::connStatusCallback %s number of elements (%d) is greater than max (%d)\n",_epicsName, _nelements, _maxElements);
#endif
      _nelements = _maxElements;
    }

    _proxy.connected(true);

    if (_monitor && !_monitored) {
      // establish monitoring
      int st;
      st = ca_create_subscription(_type,
				  _nelements,
				  _epicsChanID,
				  DBE_VALUE,
				  GetDataCallback,
				  this,
				  &_event);
      if (st != ECA_NORMAL)
        printf("%s : %s [connStatusCallback]\n", _epicsName, ca_message(st));
      else
        _monitored = true; // indicate that monitor is established so reconnects don't create additional ones
    }
  }
  else {
    printf("EpicsCAChannel::connStatusCallback %s disconnected (%p)\n",_epicsName, this);
    _connected = NotConnected;
    _proxy.connected(false);
  }
}

void EpicsCAChannel::getDataCallback(struct event_handler_args ehArgs)
{
#ifdef DBUG
  printf("EpicsCAChannel::getDataCallback[%s]\n",_epicsName);
#endif  

  if (ehArgs.status != ECA_NORMAL)
    printf("%s : %s [getDataCallback ehArgs]\n",_epicsName, ca_message(ehArgs.status));
  else {
    _proxy.getData(ehArgs.dbr);
  }
} 

void EpicsCAChannel::putDataCallback(struct event_handler_args ehArgs)
{
  if (ehArgs.status != ECA_NORMAL)
    printf("EpicsCAChannel::putDataCallback %s : %s\n",_epicsName, ca_message(ehArgs.status));
  _proxy.putStatus(ehArgs.status==ECA_NORMAL);
}

