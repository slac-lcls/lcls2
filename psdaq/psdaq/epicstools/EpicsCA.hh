#ifndef Pds_EpicsCA_hh
#define Pds_EpicsCA_hh

//--------------------------------------------------------------------------
//
// File and Version Information:
// 	$Id: EpicsCA.hh 6181 2016-07-07 01:06:10Z tookey@SLAC.STANFORD.EDU $
//
// Environment:
//      This software was developed for the BaBar collaboration.  If you
//      use all or part of it, please give an appropriate acknowledgement.
//      This class was copied from OdcCompProxy/EpicsCAProxy for use
//      in the online environment.
//
//
//------------------------------------------------------------------------

#include "psdaq/epicstools/PVMonitorCb.hh"

// epics includes
#include "cadef.h"

namespace Pds_Epics {

  class EpicsCA;

  //==============================================================================
  class EpicsCAChannel {
  public:
    enum ConnStatus { NotConnected, Connecting, Connected };

    EpicsCAChannel(const char* channelName,
                   bool        monitor,
                   EpicsCA&    proxy,
                   const int   maxElements=0);
    ~EpicsCAChannel();
    
    void connect        (void);
    void get            ();
    void put            ();
    void put_cb         ();
  public:
    void getDataCallback   (struct event_handler_args ehArgs);
    void putDataCallback   (struct event_handler_args ehArgs);
    void connStatusCallback(struct connection_handler_args chArgs);
    const char* epicsName  (void) { return _epicsName; }
    chtype     type        () const { return _type; }
    int        nelements   () const { return _nelements; }
    ConnStatus connected   () const { return _connected; }
  
  protected:
    char        _epicsName[64];
    int         _nelements;
    const int   _maxElements;
    chid	_epicsChanID;
    chtype      _type;
    evid        _event;
    ConnStatus  _connected;
    bool        _monitor;
    bool        _monitored;
    EpicsCA&    _proxy;
  };

  //==============================================================================
  class EpicsCA {
  public:
    EpicsCA(const char *channelName, PVMonitorCb*, const int maxElements=0);
    virtual ~EpicsCA();
  public:  
    virtual void  connected(bool);
    virtual void  getData  (const void* value);
    virtual void  putStatus(bool);
  public:
    unsigned sec    () const;
    unsigned nsec   () const;
    void*  data     ();
    size_t data_size() const;
    bool   connected() const;
  protected:
    EpicsCAChannel   _channel;
    PVMonitorCb*     _monitor;
    char* _pvdata;
    struct epicsTimeStamp _stamp;
    int   _pvsiz;
    bool  _connected;
    dbr_short_t _status;
    dbr_short_t _severity;
  };
};

#endif
