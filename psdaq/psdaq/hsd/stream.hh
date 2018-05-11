#ifndef HSD_STREAMHEADER_HH
#define HSD_STREAMHEADER_HH

#include "psdaq/hsd/hsd.hh"

#include <stdint.h>

namespace Pds {
  namespace HSD {
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
