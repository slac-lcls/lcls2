#ifndef Jesd204b_hh
#define Jesd204b_hh

#include "Globals.hh"

namespace Pds {
  namespace HSD {
    class Jesd204bStatus {
    public:
      unsigned gtResetDone;
      unsigned recvDataValid;
      unsigned dataNAlign;
      unsigned syncDone;
      unsigned bufOF;
      unsigned bufUF;
      unsigned commaNAlign;
      unsigned rxModEnable;
      unsigned sysRefDet;
      unsigned commaDet;
      unsigned dspErr;       // 8b
      unsigned decErr;       // 8b
      unsigned buffLatency;  // 8b
      unsigned cdrStatus;
    };

    class Jesd204b {
    public:
      static void dumpStatus(const Jesd204bStatus*,int);
    public:
      Jesd204bStatus status(unsigned) const;
    public:
      void clearErrors();
    public:
      vuint32_t reg[256];
      //  0.0  enableRx
      //  1.0  sysRefDlyRx
      //  2.0  rxPolarity
      //  4.0  subClass
      //  4.1  replEnable
      //  4.2  gtReset
      //  4.3  statClear
      //  4.4  invertSync
      //  4.5  enableScrambler
      //  5    linkErrMask
      //  5.0    alignErr
      //  5.1    decErr
      //  5.2    dspErr
      //  5.3    bufUF
      //  5.4    bufOF
      //  5.5    positionErr
      //  6.0  invertData
      //  9.0  (rxPowerDown - read/write swapped?)
      //  A.0  sysRefPeriodMin
      //  A.16 sysRefPeriodMax
      // 10-1F statusRxArr
      //   .0  gtResetDone
      //   .1  recvDataValid
      //   .2  dataNAlign
      //   .3  syncStatus
      //   .4  bufOF
      //   .5  bufUF
      //   .6  commaNAlign
      //   .7  rxModEnable
      //   .8  sysRefDet
      //   .9  commaDet
      //   .10 dspErr
      //   .18 decErr
      //   .26 buffLatency
      //   .34 cdrStatus
      // 20-2F testTXItf
      // 30-3F testSigThr
      // 40-4F statusCnt
      // 50-5F rawData
      // 60-6F statusRxArr(:32)
    };
  };
};

#endif
