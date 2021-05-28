#ifndef HSD_FexCfg_hh
#define HSD_FexCfg_hh

#include "Globals.hh"

namespace Pds {
  namespace HSD {
    class FexCfg {
    public:
      void disable();
    public:
      vuint32_t _streams;
      vuint32_t _oflow;
      vuint32_t _flowstatus;
      vuint32_t _flowidxs;

      class StreamBase {
      public:
        StreamBase() {}
      public:
        void setGate(unsigned begin, unsigned length) { 
          _reg[0] = begin;
          _reg[1] = (_reg[1] & ~0xfffff) | (length & 0xfffff);
        }
        void setFull(unsigned rows, unsigned events) { 
          _reg[2] = (rows&0xffff) | (events<<16); 
        }
        void getFree(unsigned& rows, unsigned& events, unsigned& oflow) {
#if 1   // machine check exception
          unsigned v = _reg[3];
#else
          unsigned v = 0;
#endif 
          rows   = (v>> 0)&0xffff;
          events = (v>>16)&0x1f;
          oflow  = (v>>24)&0xff;
        }
        void setPrescale(unsigned prescale) {
          _reg[1] = (_reg[1] & 0xfffff) | (prescale<<20);
        }
      public:
        vuint32_t _reg[4];
      } _base  [4];

      vuint32_t _rsvd_50[0xb0>>2];
      // vuint32_t _bram_wr_errors;
      // vuint32_t _bram_wr_sample;
      // vuint32_t _bram_rd_errors;
      // vuint32_t _bram_rd_sample;

      // vuint32_t _rsvd_80[0x80>>2];

      class Stream {
      public:
        Stream() {}
      public:
        vuint32_t rsvd [4];
        class Parm {
        public:
          vuint32_t v;
          vuint32_t rsvd;
        } parms[30];
      } _stream[4];
    };
  };
};

#endif
