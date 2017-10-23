#ifndef Pds_AxiVersion_hh
#define Pds_AxiVersion_hh

#include <string>
#include <stdint.h>

namespace Pds {
  namespace Mmhw {
    class AxiVersion {
    public:
      std::string buildStamp() const;
      std::string serialID() const;
    public:
      volatile uint32_t FpgaVersion; 
      volatile uint32_t ScratchPad; 
      volatile uint32_t DeviceDnaHigh; 
      volatile uint32_t DeviceDnaLow; 
      volatile uint32_t FdSerialHigh; 
      volatile uint32_t FdSerialLow; 
      volatile uint32_t MasterReset; 
      volatile uint32_t FpgaReload; 
      volatile uint32_t FpgaReloadAddress; 
      volatile uint32_t Counter; 
      volatile uint32_t FpgaReloadHalt; 
      volatile uint32_t reserved_11[0x100-11];
      volatile uint32_t UserConstants[64];
      volatile uint32_t reserved_0x140[0x1c0-0x140];
      volatile uint32_t dnaValue[4];
      volatile uint32_t reserved_0x1c4[0x200-0x1c4];
      volatile uint32_t BuildStamp[64];
    };
  };
};

#endif
