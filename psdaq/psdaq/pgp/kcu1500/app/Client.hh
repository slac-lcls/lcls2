#ifndef Pds_Kcu_Client_hh
#define Pds_Kcu_Client_hh

#include "xtcdata/xtc/Dgram.hh"

#include <time.h>
#include <stdint.h>

namespace Pds {
  namespace Kcu {
    class Client {
    public:
      Client(const char* devname);
      ~Client();
    public:
      //  Enable the trigger
      void start(unsigned partn);
      //  Disable the trigger
      void stop();
      //  Set read retry interval and max attempts
      void set_retry(unsigned intv_us, unsigned retries);
      //
      //  Seek timing frame by pulseId (if non-zero)
      //  Return address on success or 0 on failure
      //
      const XtcData::Transition* advance(uint64_t pulseId=0);

      void dump();
    private:
      int               _fd;
      void **           _dmaBuffers;
      int               _current;
      int               _ret;
      unsigned          _retry_intv;
      unsigned          _retry_num;
      uint64_t          _next;
      XtcData::Transition _tr;
      enum { max_ret_cnt = 10000 };
      uint32_t*         _rxFlags;
      uint32_t*         _dest;
      uint32_t*         _dmaIndex;
      int32_t*          _dmaRet;
      unsigned          _skips;
      unsigned          _retries;
    };
  };
};

#endif
