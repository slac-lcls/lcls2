#ifndef Pds_TriggerEventManager_hh
#define Pds_TriggerEventManager_hh

#include <stdint.h>

namespace Pds {
  namespace Mmhw {
    class XpmMessageAligner {
    public:
      XpmMessageAligner() {}
    public:
      uint32_t messageDelay[8];
      uint32_t txId;
      uint32_t rxId;
      uint32_t reserved_28[(0x100-0x28)>>2];
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
      uint32_t enable;
      uint32_t group;
      uint32_t pauseThresh;
      uint32_t triggerDelay;
      uint32_t status;
      uint32_t l0Count;
      uint32_t l1AcceptCount;
      uint32_t l1RejectCount;
      uint32_t transitionCount;
      uint32_t validCount;
      uint32_t triggerCount;
      uint32_t currPartitionBcast;
      uint64_t currPartitionWord0;
      uint32_t fullToTrig;
      uint32_t nfullToTrig;
      uint32_t resetCounters;
      uint32_t reserved_44[(0x100-0x44)>>2];
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
