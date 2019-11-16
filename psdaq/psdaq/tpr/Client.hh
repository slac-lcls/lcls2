#ifndef Pds_Tpr_Client_hh
#define Pds_Tpr_Client_hh

#include "Module.hh"

#include <stdint.h>

namespace Pds {
  namespace Tpr {
    class Frame;
    class Queues;
    class TprReg;

    class Client {
    public:
      Client(const char* devname,
             unsigned    channel = 0);
      ~Client();
    public:
      //  Enable the trigger
      void start(unsigned partn);
      //  Enable the trigger (full rate)
      void start(TprBase::FixedRate rate=TprBase::FixedRate::_1M);
      //  Disable the trigger
      void stop();
      //  Release control
      void release(); 
      //
      //  Seek timing frame by pulseId
      //  Return address on success or 0
      //
      const Pds::Tpr::Frame* advance(uint64_t pulseId);
      //
      //  Return next timing frame
      //
      const Pds::Tpr::Frame* advance();
    private:
      void _dump() const;
    private:
      unsigned          _channel;
      int               _fd;
      Pds::Tpr::TprReg* _dev;
      int               _fdsh;
      Pds::Tpr::Queues* _queues;
      unsigned          _rp;
    };
  };
};

#endif
