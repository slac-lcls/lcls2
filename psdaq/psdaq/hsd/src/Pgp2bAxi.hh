#ifndef HSD_PGP2BAXI_HH
#define HSD_PGP2BAXI_HH

#include "Globals.hh"

namespace Pds {
  namespace HSD {
    class Pgp2bAxi {
    public:
      void dump() const;
    public:
      vuint32_t _countReset;
      vuint32_t _rxReset;
      vuint32_t _flush;
      vuint32_t _loopback;
      vuint32_t _txUserData;
      vuint32_t _autoStatusSendEnable;
      vuint32_t _disableFlowControl;
      vuint32_t _reserved_0x1c;
      vuint32_t _status;
      vuint32_t _remoteUserData;
      vuint32_t _cellErrCount;
      vuint32_t _linkDownCount;
      vuint32_t _linkErrCount;
      vuint32_t _remoteOvfVc0;
      vuint32_t _remoteOvfVc1;
      vuint32_t _remoteOvfVc2;
      vuint32_t _remoteOvfVc3;
      vuint32_t _rxFrameErrs;
      vuint32_t _rxFrames;
      vuint32_t _localOvfVc0;
      vuint32_t _localOvfVc1;
      vuint32_t _localOvfVc2;
      vuint32_t _localOvfVc3;
      vuint32_t _txFrameErrs;
      vuint32_t _txFrames;
      vuint32_t _rxClkFreq;
      vuint32_t _txClkFreq;
      vuint32_t _reserved_0x6c;
      vuint32_t _lastTxOpcode;
      vuint32_t _lastRxOpcode;
      vuint32_t _txOpcodes;
      vuint32_t _rxOpcodes;
      vuint32_t _reserved[32];
    };
  }
}

#endif
