/**
 * @file
 * @brief MmcmPhaseLock, the register block of an MMCM phase-lock unit of the XPM.
 */
#ifndef Pds_Xpm_MmcmPhaseLock_hh
#define Pds_Xpm_MmcmPhaseLock_hh

#include "psdaq/cphw/Reg.hh"

namespace Pds {
  namespace Xpm {
    /** Register block (1 MiB of address space) of an MMCM phase-lock unit; register meanings below come from their names and their use in PVCtrls.cc. */
    class MmcmPhaseLock {
    public:
      /** Return true if bit 30 of delayValue is clear. */
      bool ready() const { return (delayValue & (1<<30))==0; }
      /** Write 1 to the reset register. */
      void reset() { _reset = 1; }
    public:
      Cphw::Reg delaySet;  ///< Register named delaySet; not used by the C++ code in psdaq (psdaq/psdaq/pyxpm writes a register of the same name in its own Python device model). Inferred from the name; not verified.
      Cphw::Reg delayValue;  ///< Register named delayValue; ready() tests its bit 30 and PVCtrls.cc reads it.
      Cphw::Reg ramAddr;  ///< Register named ramAddr; PVCtrls.cc writes an index to it before reading ramData.
      Cphw::Reg ramData;  ///< Register named ramData; PVCtrls.cc reads it after setting ramAddr.
    private:
      Cphw::Reg _reset;
      uint32_t rsvd[(0x00100000-20)>>2];
    };
  };
};

#endif
