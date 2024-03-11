#ifndef Pds_TriggerEventManager_hh
#define Pds_TriggerEventManager_hh

#include "psdaq/mmhw/Reg.hh"
#include <stdint.h>

namespace Pds {
  namespace Mmhw {
    class XpmMessageAligner {
    public:
      XpmMessageAligner() {}
    public:
      Reg      messageDelay[8];
      Reg      txId;
      Reg      rxId;
      Reg      reserved_28[(0x100-0x28)>>2];
    };

    class TriggerEventBuffer {
    public:
      TriggerEventBuffer() {}
    public:
      void start  (unsigned group,
                   unsigned triggerDelay=10,
                   unsigned pauseThresh=16);
      void stop   ();
    public:
      Reg      enable;
      Reg      group;
      Reg      pauseThresh;
      Reg      triggerDelay;
      Reg      status;
      Reg      l0Count;
      Reg      l1AcceptCount;
      Reg      l1RejectCount;
      Reg      transitionCount;
      Reg      validCount;
      Reg      triggerCount;
      Reg      currPartitionBcast;
      uint64_t currPartitionWord0;
      Reg      fullToTrig;
      Reg      nfullToTrig;
      Reg      resetCounters;
      Reg      reserved_44[(0x100-0x44)>>2];
    };

    class TriggerEventManager {
    public:
      TriggerEventManager() {}
    public:
      XpmMessageAligner&  xma() { return _xma; }
      TriggerEventBuffer& det(unsigned i) { return reinterpret_cast<TriggerEventBuffer*>(this+1)[i]; }
    private:
      XpmMessageAligner _xma;
    };
  };
};

#endif
