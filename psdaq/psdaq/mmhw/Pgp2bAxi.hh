#ifndef PDS_PGP2BAXI_HH
#define PDS_PGP2BAXI_HH

#include "psdaq/mmhw/Reg.hh"
#include <unistd.h>
#include <stdint.h>

namespace Pds {
  namespace Mmhw {
    class Pgp2bAxi {
    public:
      void dump() const;
    public:
      Reg _countReset;
      Reg _rxReset;
      Reg _flush;
      Reg _loopback;
      Reg _txUserData;
      Reg _autoStatusSendEnable;
      Reg _disableFlowControl;
      Reg _reserved_0x1c;
      Reg _status;
      Reg _remoteUserData;
      Reg _cellErrCount;
      Reg _linkDownCount;
      Reg _linkErrCount;
      Reg _remoteOvfVc0;
      Reg _remoteOvfVc1;
      Reg _remoteOvfVc2;
      Reg _remoteOvfVc3;
      Reg _rxFrameErrs;
      Reg _rxFrames;
      Reg _localOvfVc0;
      Reg _localOvfVc1;
      Reg _localOvfVc2;
      Reg _localOvfVc3;
      Reg _txFrameErrs;
      Reg _txFrames;
      Reg _rxClkFreq;
      Reg _txClkFreq;
      Reg _reserved_0x6c;
      Reg _lastTxOpcode;
      Reg _lastRxOpcode;
      Reg _txOpcodes;
      Reg _rxOpcodes;
      Reg _reserved[32];
    };
  }
}

#endif
