/**
 * @file
 * @brief Declares psalg::shmem::ProcInfo, a Src subclass whose process-id and IP accessors are stubs.
 */
#ifndef PsAlg_ShMem_ProcInfo_hh
#define PsAlg_ShMem_ProcInfo_hh

#include <stdint.h>
#include "xtcdata/xtc/Src.hh"
#include "xtcdata/xtc/Level.hh"

using namespace XtcData;

namespace psalg {
  namespace shmem {
    // For all levels except Source

    /** Src subclass with process-id and IP-address accessors that are stubs in ProcInfo.cc (only the level is kept). The comment above says it is for all levels except Source. */
    class ProcInfo : public Src {
    public:

  //     ProcInfo();
      /** Construct as Src(level), with value bits 0; processId and ipAddr are ignored. */
      ProcInfo(Level::Type level, uint32_t processId, uint32_t ipAddr);

      /** Stub: always returns 0. */
      uint32_t processId() const;
      /** Stub: always returns 0. */
      uint32_t ipAddr()    const;
      /** Stub: does nothing. */
      void     ipAddr(int);
    };
  }
}
#endif
