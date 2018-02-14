#ifndef HSD_PGP2BAXI_HH
#define HSD_PGP2BAXI_HH

#include <unistd.h>
#include <stdint.h>

namespace Pds {
  namespace HSD {
    class Pgp2bAxi {
    public:
      void dump() const;
    public:
      uint32_t _countReset;
      uint32_t _rxReset;
      uint32_t _flush;
      uint32_t _loopback;
      uint32_t _txUserData;
      uint32_t _autoStatusSendEnable;
      uint32_t _disableFlowControl;
      uint32_t _reserved_0x1c;
      uint32_t _status;
      uint32_t _remoteUserData;
      uint32_t _cellErrCount;
      uint32_t _linkDownCount;
      uint32_t _linkErrCount;
      uint32_t _remoteOvfVc0;
      uint32_t _remoteOvfVc1;
      uint32_t _remoteOvfVc2;
      uint32_t _remoteOvfVc3;
      uint32_t _rxFrameErrs;
      uint32_t _rxFrames;
      uint32_t _localOvfVc0;
      uint32_t _localOvfVc1;
      uint32_t _localOvfVc2;
      uint32_t _localOvfVc3;
      uint32_t _txFrameErrs;
      uint32_t _txFrames;
      uint32_t _rxClkFreq;
      uint32_t _txClkFreq;
      uint32_t _reserved_0x6c;
      uint32_t _lastTxOpcode;
      uint32_t _lastRxOpcode;
      uint32_t _txOpcodes;
      uint32_t _rxOpcodes;
      uint32_t _reserved[32];
    };
  }
}

#endif
