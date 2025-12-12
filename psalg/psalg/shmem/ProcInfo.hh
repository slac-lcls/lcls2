#ifndef PsAlg_ShMem_ProcInfo_hh
#define PsAlg_ShMem_ProcInfo_hh

#include <stdint.h>
#include "xtcdata/xtc/Src.hh"
#include "xtcdata/xtc/Level.hh"

using namespace XtcData;

namespace psalg {
  namespace shmem {
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
}
#endif
