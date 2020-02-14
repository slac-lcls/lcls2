#ifndef HSD_FexCfg_hh
#define HSD_FexCfg_hh

#include "Globals.hh"

namespace Pds {
  namespace HSD {
    class FexCfg {
    public:
      vuint32_t _streams;
      vuint32_t _oflow;
      // vuint32_t _flowstatus;
      // vuint32_t _flowidxs;
      uint32_t  _rsvd_08[2];

      class StreamBase {
      public:
        StreamBase() {}
      public:
        void setGate(unsigned begin, unsigned length) { _gate = (begin&0xffff) | (length<<16); }
        void setFull(unsigned rows, unsigned events) { _full = (rows&0xffff) | (events<<16); }
        void getFree(unsigned& rows, unsigned& events) {
          unsigned v = _free;
          rows   = v&0xffff;
          events = (v>>16)&0xff;
        }
      public:
        vuint32_t _prescale;
        vuint32_t _gate;
        vuint32_t _full; 
        vuint32_t _free;
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
