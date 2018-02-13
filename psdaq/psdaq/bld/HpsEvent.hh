#ifndef HpsEvent_hh
#define HpsEvent_hh

#include <vector>
#include <stdint.h>

namespace Bld {
  class HpsEvent {
  public:
    uint64_t  timeStamp;
    uint64_t  pulseId;
    uint32_t  mask;
    uint32_t  beam;
    std::vector<uint32_t> channels;
    uint32_t  valid;
  };
};
#endif
