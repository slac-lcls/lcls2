#ifndef HSD_STREAMHEADER_HH
#define HSD_STREAMHEADER_HH

#include "xtcdata/xtc/Dgram.hh"
#include <stdint.h>
#include <stdio.h>

namespace Pds {
  namespace HSD {

  class EventHeader : public XtcData::L1Dgram {
    public:
      EventHeader() {}
    public:
      uint64_t pulseId   () const { return seq.pulseId().value(); }
      unsigned eventType () const { return seq.pulseId().control(); }
      uint64_t timeStamp () const { return seq.stamp().value(); }//nanoseconds() | (uint64_t)(seq.stamp().seconds())<<32; }
      unsigned samples   () const { return _info[0]&0xfffff; }
      unsigned streams   () const { return (_info[0]>>20)&0xf; }
      unsigned channels  () const { return (_info[0]>>24)&0xff; }
      unsigned sync      () const { return _info[1]&0x7; }

      void dump() const
      {
        printf("EventHeader dump\n");
        uint32_t* word = (uint32_t*) this;
        for(unsigned i=0; i<8; i++)
          printf("[%d] %08x ", i, word[i]);//, i<7 ? '.' : '\n');
        printf("pID [%016llx]  time [%u.%09u]  trig [%04x]  sync [%u]\n",
               (unsigned long long) pulseId(), seq.stamp().seconds(), seq.stamp().nanoseconds(),
               readoutGroups(), sync());
        printf("####@ 0x%x 0x%x 0x%x %u %u %u %llu\n", env, _info[0], _info[1], samples(), streams(), channels(), (unsigned long long) timeStamp());
      }
  private:
      uint32_t _info[2];
  };

  class StreamHeader {
    public:
      StreamHeader() {}
    public:
      unsigned num_samples() const { return _word[0]&~(1<<31); }
      unsigned stream_id  () const { return (_word[1]>>24)&0xff; }
      unsigned samples () const { return _word[0]&0x7fffffff; } // number of samples
      bool     overflow() const { return _word[0]>>31; }        // overflow of memory buffer
      unsigned strmtype() const { return (_word[1]>>24)&0xff; } // type of stream {raw, thr, ...}
      unsigned boffs   () const { return (_word[1]>>0)&0xff; }  // padding at start
      unsigned eoffs   () const { return (_word[1]>>8)&0xff; }  // padding at end
      unsigned buffer  () const { return _word[1]>>16; }        // 16 front-end buffers (like FEE)
      // (only need 4 bits but using 16)
      unsigned toffs   () const { return _word[2]; }            // phase between sample clock and timing clock (1.25GHz)
      // wrong if this value is not fixed
      unsigned baddr   () const { return _word[3]&0xffff; }     // begin address in circular buffer
      unsigned eaddr   () const { return _word[3]>>16; }        // end address in circular buffer
      void     dump    () const
      {
        printf("StreamHeader dump\n");
        printf("  ");
        for(unsigned i=0; i<4; i++)
          printf("%08x%c", _word[i], i<3 ? '.' : '\n');
        printf("  size [%04u]  boffs [%u]  eoffs [%u]  buff [%u]  toffs[%04u]  baddr [%04x]  eaddr [%04x]\n",
               samples(), boffs(), eoffs(), buffer(), toffs(), baddr(), eaddr());
      }
    private:
      uint32_t _word[4];
  };

  class RawStream {
    public:
      RawStream(const EventHeader& event, const StreamHeader& strm);
    public:
      static void interleave(bool v);
      static void verbose   (unsigned);

      bool validate(const EventHeader& event, const StreamHeader& next) const;
    private:
      unsigned adcVal(uint64_t pulseId) const;
      uint16_t next(uint16_t adc) const;
    private:
      uint64_t _pid;
      unsigned _adc;
      unsigned _baddr;
      unsigned _eaddr;
  };


    //
    //  Validate threshold stream : ramp signal repeats 0..0xfe
    //      phyclk period is 0.8 ns
    //      recTimingClk period is 5.384 ns
    //        => 1346 phyclks per beam period
    //

    class ThrStream {
    public:
      ThrStream(const StreamHeader& strm);
    public:
      bool validate(const StreamHeader& raw) const;
    private:
      const StreamHeader& _strm;
    };

  }
}

#endif
