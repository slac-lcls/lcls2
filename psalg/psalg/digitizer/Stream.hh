/**
 * @file
 * @brief Declares HSD digitizer helpers: the StreamHeader and EventHeader views and the RawStream and ThrStream validators.
 */
#ifndef HSD_STREAMHEADER_HH
#define HSD_STREAMHEADER_HH

#include "xtcdata/xtc/Dgram.hh"
#include <stdint.h>
#include <stdio.h>

/** Outer namespace of the HSD digitizer helpers. */
namespace Pds {
  /** HSD digitizer classes: stream and event header views and stream validators. */
  namespace HSD {

  // used by class Channel

  /**
   * View of a 4-word (4 x uint32_t) stream header; the stream's uint16_t samples follow it in memory, as read by RawStream and ThrStream validate().
   * The accessors decode bit fields of the four words. The comment above says it is used by class Channel.
   */
  class StreamHeader {
    public:
      /** Default constructor; the words are left uninitialized. */
      StreamHeader() {}
    public:
      /** Return word 0 bits 28:0. */
      unsigned num_samples() const { return _word[0]&0x1fffffff; }
      /** Return word 1 bits 31:24. */
      unsigned stream_id  () const { return (_word[1]>>24)&0xff; }
      /** Return num_samples() (word 0 bits 28:0). */
      unsigned samples () const { return num_samples(); } // number of samples
      /** Return word 0 bit 29; the trailing comment says sample upper bits were dropped. */
      bool     out_of_range() const { return (_word[0]>>29)&1; }    // sample upper bits dropped
      /** Return word 0 bit 30; the trailing comment says the data serial link was unlocked. */
      bool     unlocked() const { return (_word[0]>>30)&1; }        // data serial link unlocked
      /** Return word 0 bit 31; the trailing comment says the memory buffer overflowed. */
      bool     overflow() const { return (_word[0]>>31)&1; }        // overflow of memory buffer
      /** Return word 1 bits 31:24 (the same bits as stream_id()); the trailing comment calls it the stream type (raw, thr, ...). */
      unsigned strmtype() const { return (_word[1]>>24)&0xff; } // type of stream {raw, thr, ...}
      /** Return word 1 bits 7:0; the trailing comment calls it padding at the start. The validators start reading samples at this index. */
      unsigned boffs   () const { return (_word[1]>>0)&0xff; }  // padding at start
      /** Return word 1 bits 15:8; the trailing comment calls it padding at the end. */
      unsigned eoffs   () const { return (_word[1]>>8)&0xff; }  // padding at end
      /** Return word 1 bits 31:16 (overlapping stream_id()); the comments say it is one of 16 front-end buffers and only 4 bits are needed. */
      unsigned buffer  () const { return _word[1]>>16; }        // 16 front-end buffers (like FEE)
      // (only need 4 bits but using 16)
      /** Return word 2 bits 15:0; the trailing comment calls it the phase between the sample clock and the timing clock. */
      unsigned toffs   () const { return (_word[2]>> 0)&0xffff; } // phase between sample clock and timing clock (1.25GHz)
      /** Return word 2 bits 20:16; the trailing comment calls it the trigger tag word. */
      unsigned l1tag   () const { return (_word[2]>>16)&0x1f; }   // trigger tag word
      // wrong if this value is not fixed
      /** Return word 3 bits 15:0; the trailing comment calls it the begin address in the circular buffer. */
      unsigned baddr   () const { return _word[3]&0xffff; }     // begin address in circular buffer
      /** Return word 3 bits 31:16; the trailing comment calls it the end address in the circular buffer. */
      unsigned eaddr   () const { return _word[3]>>16; }        // end address in circular buffer
      /** Print the four words in hex and the decoded samples(), boffs(), eoffs(), buffer(), toffs(), baddr() and eaddr() to stdout. */
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

  // this class is used only for testing (psalg/tests/hsd_valid.cc)

  /** L1Dgram followed by two 32-bit info words decoded by samples(), streams(), channels() and sync(). The comment above says it is used only for testing (psalg/tests/hsd_valid.cc). */
  class EventHeader : public XtcData::L1Dgram {
    public:
      /** Default constructor; nothing is initialized. */
      EventHeader() {}
    public:
      /** Return service(), the transition id from the env word. */
      unsigned eventType () const { return service(); }
      /** Return time.value() (seconds in the high 32 bits, nanoseconds in the low 32 bits). */
      uint64_t timeStamp () const { return time.value(); }
      /** Return info word 0 bits 19:0. */
      unsigned samples   () const { return _info[0]&0xfffff; }
      /** Return info word 0 bits 23:20. */
      unsigned streams   () const { return (_info[0]>>20)&0xf; }
      /** Return info word 0 bits 31:24. */
      unsigned channels  () const { return (_info[0]>>24)&0xff; }
      /** Return info word 1 bits 2:0. */
      unsigned sync      () const { return _info[1]&0x7; }

      /** Print to stdout the first eight 32-bit words of the object in hex, the time, readoutGroups() and sync(), and a line with env, both info words and the decoded fields. */
      void dump() const
      {
        printf("EventHeader dump\n");
        uint32_t* word = (uint32_t*) this;
        for(unsigned i=0; i<8; i++)
          printf("[%d] %08x ", i, word[i]);//, i<7 ? '.' : '\n');
        printf("time [%u.%09u]  trig [%04x]  sync [%u]\n",
               time.seconds(), time.nanoseconds(),
               readoutGroups(), sync());
        printf("####@ 0x%x 0x%x 0x%x %u %u %u %llu\n", env, _info[0], _info[1], samples(), streams(), channels(), (unsigned long long) timeStamp());
      }
  private:
      uint32_t _info[2];
  };

  // this class is used only for testing (psalg/tests/hsd_valid.cc)

  /** Validator that expects each raw sample to be the previous sample plus 1 modulo 0x800. The comment above says it is used only for testing; its constructor always throws. */
  class RawStream {
    public:
      /** Store the first sample and address range of strm, then print a message and throw a const char* (pulse-id support was removed), so construction never succeeds. event is not used. */
      RawStream(const EventHeader& event, const StreamHeader& strm);
    public:
      /** Set the file-static interleave flag read by the private adcVal(). With the pulse id fixed at 0 it does not change the result. */
      static void interleave(bool v);
      /** Set the file-static verbosity used by RawStream::validate() and ThrStream::validate(): nonzero prints mismatches, and above 1 RawStream::validate() also prints the error count. */
      static void verbose   (unsigned);

      /**
       * Return true if next has no samples, or if every sample from boffs() up to samples()-eoffs() matches: the first equals the stored first sample (masked to 11 bits) and each later one equals the previous sample plus 1, masked with 0x7ff.
       * event is not used.
       */
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

    /** Validator that compares a threshold (compressed) stream with the matching raw stream. The comment above describes a ramp test signal repeating 0..0xfe. */
    class ThrStream {
    public:
      /** Keep a reference to the threshold stream header strm (not a copy). */
      ThrStream(const StreamHeader& strm);
    public:
      /**
       * Return true if either stream has no samples, or if every non-skip word of the threshold stream equals the raw sample at the matching position. The comparison starts at each stream's boffs() and stops at whichever stream ends first (its samples() minus eoffs()); words past that point are not checked.
       * Words with bit 15 set are skip markers that advance the raw position by their low 15 bits; a skip marker in the first position advances both positions by one. When both streams have samples, prints the error and test counts to stdout (nothing is printed on the early return for an empty stream).
       */
      bool validate(const StreamHeader& raw) const;
    private:
      const StreamHeader& _strm;
    };

  }
}

#endif
