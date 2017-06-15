#ifndef Pds_ProcInfo_hh
#define Pds_ProcInfo_hh

#include <stdint.h>
#include "pdsdata/xtc/Src.hh"
#include "pdsdata/xtc/Level.hh"

namespace Pds {

  // For all levels except Source

  class ProcInfo : public Src {
  public:

//     ProcInfo();
    ProcInfo(Level::Type level, uint32_t processId, uint32_t ipAddr);

    uint32_t processId() const;
    uint32_t ipAddr()    const;
    void     ipAddr(int);
  };

}
#endif
