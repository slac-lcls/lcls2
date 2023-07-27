#ifndef HpsEvent_hh
#define HpsEvent_hh

#include <vector>
#include <stdint.h>

namespace Bld {
  class HpsEvent {
  public:
    uint64_t  timeStamp;
    uint64_t  pulseId;
    uint32_t  beam;
    uint32_t  channels;
    uint64_t  sevr;
    //    uint32_t channels[];
  public:
    const uint32_t* channelData() const { return reinterpret_cast<const uint32_t*>(this+1); }
  };
};
#endif
