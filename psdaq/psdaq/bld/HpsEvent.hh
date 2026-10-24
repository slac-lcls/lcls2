/**
 * @file
 * @brief Bld::HpsEvent, one event decoded from an HPS BLD packet by HpsEventIterator.
 */
#ifndef HpsEvent_hh
#define HpsEvent_hh

#include <vector>
#include <stdint.h>

/** Namespace of the HPS BLD packet decoding classes (HpsEvent, HpsEventIterator). */
namespace Bld {
  /** One event decoded from an HPS BLD packet; the channel values follow the object in memory (see channelData()). */
  class HpsEvent {
  public:
    uint64_t  timeStamp;  ///< Time stamp; HpsEventIterator adds the 20-bit offset of later events to that of the first.
    uint64_t  pulseId;  ///< Pulse ID; HpsEventIterator adds the 12-bit offset of later events to that of the first.
    uint32_t  beam;  ///< 32-bit beam word of the event, copied from the packet. Inferred from the name; not verified.
    uint32_t  channels;  ///< Channel mask from the first event of the packet; one value follows for each set bit.
    uint64_t  sevr;  ///< 64-bit severity word of the event, copied from the packet. Inferred from the name; not verified.
    //    uint32_t channels[];
  public:
    /** Return the channel values stored right after this object. */
    const uint32_t* channelData() const { return reinterpret_cast<const uint32_t*>(this+1); }
  };
};
#endif
