#ifndef Pds_Src_hh
#define Pds_Src_hh

#include <stdint.h>
#include "pdsdata/xtc/Level.hh"

namespace Pds {

  class Node;

  class Src {
  public:

    Src();
    Src(Level::Type level);

    uint32_t log()   const;
    uint32_t phy()   const;

    Level::Type level() const;

    bool operator==(const Src& s) const;
    bool operator<(const Src& s) const;

    static uint32_t _sizeof() { return sizeof(Src); }
  protected:
    uint32_t _log; // logical  identifier
    uint32_t _phy; // physical identifier
  };

}
#endif
