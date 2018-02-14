#ifndef HSD_FexCfg_hh
#define HSD_FexCfg_hh

#include <stdint.h>

namespace Pds {
  namespace HSD {
    class FexCfg {
    public:
      uint32_t _streams;
      uint32_t _rsvd[3];

      class StreamBase {
      public:
        StreamBase() {}
      public:
        void setGate(unsigned begin, unsigned length) { _gate = (begin&0xffff) | (length<<16); }
        void setFull(unsigned rows, unsigned events) { _full = (rows&0xffff) | (events<<16); }
        void getFree(unsigned& rows, unsigned& events) {
          unsigned v = _free;
          rows   = v&0xffff;
          events = v>>16;
        }
      public:
        uint32_t _prescale;
        uint32_t _gate;
        uint32_t _full; 
        uint32_t _free;
      } _base  [4];

      uint32_t _rsvd2[44];

      class Stream {
      public:
        Stream() {}
      public:
        uint32_t rsvd [4];
        class Parm {
        public:
          uint32_t v;
          uint32_t rsvd;
        } parms[30];
      } _stream[4];

    private:
      uint32_t _rsvd3[(0x1000-0x500)>>2];
    };
  };
};

#endif
