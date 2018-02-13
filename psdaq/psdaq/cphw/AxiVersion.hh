#ifndef Pds_Cphw_AxiVersion_hh
#define Pds_Cphw_AxiVersion_hh

#include "psdaq/cphw/Reg.hh"

#include <string>

namespace Pds {
  namespace Cphw {
    class AxiVersion {
    public:
      std::string buildStamp() const;
    public:
      Reg FpgaVersion; 
      Reg ScratchPad; 
      Reg DeviceDnaHigh; 
      Reg DeviceDnaLow; 
      Reg FdSerialHigh; 
      Reg FdSerialLow; 
      Reg MasterReset; 
      Reg FpgaReload; 
      Reg FpgaReloadAddress; 
      Reg Counter; 
      Reg FpgaReloadHalt; 
      Reg reserved_11[0x100-11];
      Reg UserConstants[64];
      Reg reserved_0x140[0x200-0x140];
      Reg BuildStamp[64];
      Reg reserved_0x240[0x4000-0x240];
    };
  }
}

#endif
