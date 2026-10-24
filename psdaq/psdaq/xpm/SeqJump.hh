/**
 * @file
 * @brief SeqJump, the jump registers of an XPM sequence engine.
 */
#ifndef SeqJump_hh
#define SeqJump_hh

#include "psdaq/cphw/Reg.hh"

namespace Pds {
  namespace Xpm {
    /** Jump registers of one sequence engine (16 words); only register 15 is used here. */
    class SeqJump {
    public:
      /** Does nothing; empty body. */
      SeqJump() {}
    public:
      /** Set the upper 16 bits of register 15 to sync. */
      void setManSync (unsigned sync) { 
        unsigned r = _reg[15];
        r &= ~0xffff0000;
        r |= (0xffff0000 & (sync<<16));
        _reg[15] = r;
      }
      /** Set the lower 16 bits of register 15 to addr (12 bits) and pclass (4 bits, above addr). */
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
