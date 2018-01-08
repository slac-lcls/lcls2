#ifndef HSD_EVENTHEADER_HH
#define HSD_EVENTHEADER_HH

#include <stdint.h>

namespace Pds {
  namespace HSD {

    class EventHeader : public XtcData::L1Transition {
    public:
      EventHeader() {}
    public:
      uint64_t pulseId   () const { return seq.stamp().pulseId(); }
      uint64_t timeStamp () const { return seq.clock().nanoseconds() | (uint64_t)(seq.clock().seconds())<<32; }
      uint32_t eventCount() const { return evtcounter; }
      unsigned samples   () const { return (env>>32)&0xfffff; }
      unsigned streams   () const { return (env>>52)&0xf; }
      unsigned channels  () const { return (env>>56)&0xff; }
      unsigned sync      () const { return _syncword&0x7; }

      void dump()
      {
        uint32_t* word = (uint32_t*) this;
        for(unsigned i=0; i<8; i++)
          printf("%08x%c", word[i], i<7 ? '.' : '\n');
        printf("pID [%016lux]  time [%u.%09u]  trig [%04x]  event [%u]  sync [%u]\n",
               pulseId(), seq.clock().seconds(), seq.clock().nanoseconds(),
               readoutGroups(), eventCount(), sync());
      }
    private:
      uint32_t _syncword;
    };

  }
}

#endif
