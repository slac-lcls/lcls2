#ifndef Pds_Tpr_Client_hh
#define Pds_Tpr_Client_hh

#include <stdint.h>

namespace Pds {
  namespace Tpr {
    class Frame;
    class Queues;
    class TprReg;

    class Client {
    public:
      Client(const char* devname);
      ~Client();
    public:
      //  Enable the trigger
      void start(unsigned partn);
      //  Disable the trigger
      void stop();
      //
      //  Seek timing frame by pulseId
      //  Return address on success or 0
      //
      const Pds::Tpr::Frame* advance(uint64_t pulseId);
    private:
      int               _fd;
      Pds::Tpr::TprReg* _dev;
      int               _fdsh;
      Pds::Tpr::Queues* _queues;
      unsigned          _chnrp;
    };
  };
};

#endif
