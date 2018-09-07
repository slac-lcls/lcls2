#ifndef Pds_MonReq_XtcMonitorClient_hh
#define Pds_MonReq_XtcMonitorClient_hh

namespace XtcData {

    class Dgram;

};

namespace Pds {
  namespace MonReq {

    class XtcMonitorClient {
    public:
      XtcMonitorClient() {}
      virtual ~XtcMonitorClient() {};

    public:
      //
      //  tr_index must be unique among clients
      //  unique values of ev_index produce a serial chain of clients sharing events
      //  common values of ev_index produce a set of clients competing for events
      //
      int run(const char* tag, int tr_index=0);
      int run(const char* tag, int tr_index, int ev_index);
      virtual int processDgram(XtcData::Dgram*);
    };
  };
};
#endif
