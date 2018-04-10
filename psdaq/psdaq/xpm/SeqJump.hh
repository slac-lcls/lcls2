#ifndef SeqJump_hh
#define SeqJump_hh

#include "psdaq/cphw/Reg.hh"

namespace Pds {
  namespace Xpm {
    class SeqJump {
    public:
      SeqJump() {}
    public:
      void setManSync (unsigned sync) { 
        unsigned r = _reg[15];
        r &= ~0xffff0000;
        r |= (0xffff0000 & (sync<<16));
        _reg[15] = r;
      }
      void setManStart(unsigned addr, unsigned pclass) { 
        unsigned v = (addr&0xfff) | ((pclass&0xf)<<12);
        unsigned r = _reg[15];
        r &= ~0xffff;
        r |= (0xffff & v);
        _reg[15] = r;
      }
    private:
      Cphw::Reg _reg[16];
    };
  };
};

#endif
