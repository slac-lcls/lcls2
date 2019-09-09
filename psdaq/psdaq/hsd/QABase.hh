#ifndef HSD_QABase_hh
#define HSD_QABase_hh

#include "Globals.hh"

namespace Pds {
  namespace HSD {

    class QABase {
    public:
      void init();
      void setChannels(unsigned);
      enum Interleave { Q_NONE, Q_ABCD };
      void setMode    (Interleave);
      void setupDaq (unsigned partition);
      void setupLCLS(unsigned rate);
      void setupLCLSII(unsigned rate);
      void setTrigShift(unsigned shift);
      void enableDmaTest(bool);
      void start();
      void stop ();
      void resetCounts();
      void resetClock (bool);
      void resetDma   ();
      void resetFb    ();
      void resetFbPLL ();
      void setLocalId (unsigned v);
      void dump() const;
    public:
      void start   (unsigned fmc);
      void stop    (unsigned fmc);
      void setupDaq(unsigned partition, unsigned fmc);
      unsigned running() const;
    public:
      vuint32_t irqEnable;
      vuint32_t irqStatus;
      vuint32_t partitionAddr;
      vuint32_t dmaFullThr;
      vuint32_t csr; 
      //   CSR
      // [ 0:0]  count reset
      // [ 1:1]  dma size histogram enable (unused)
      // [ 2:2]  dma test pattern enable
      // [ 3:3]  adc sync reset
      // [ 4:4]  dma reset
      // [ 5:5]  fb phy reset
      // [ 6:6]  fb pll reset
      // [ 8:15] trigger bit mask shift
      // [31:30] acqEnable per FMC
      vuint32_t acqSelect;
      //   AcqSelect
      // [12: 0]  rateSel  :   [7:0] eventCode
      // [31:13]  destSel
      vuint32_t control;
      //   Control
      // [7:0] channel enable mask
      // [8:8] interleave
      // [19:16] partition FMC 0
      // [23:20] partition FMC 1
      // [24]  inhibit
      vuint32_t samples;       //  Must be a multiple of 16 [v1 only]
      vuint32_t prescale;      //  Sample prescaler [v1 only]
      //   Prescale
      //   Values are mapped as follows:
      //  Value    Rate Divisor    Nominal Rate
      //   [0..1]       1           1250 MHz
      //     2          2            625 MHz
      //   [3..5]       5            250 MHz
      //   [6..7]      10            125 MHz
      //   [8..15]     20             62.5 MHz
      //  [16..23]     30             41.7 MHz
      //  [24..31]     40             31.3 MHz
      //  [32..39]     50             25 MHz
      //  [40..47]     60             20.8 MHz
      //  [48..55]     70             17.9 MHz
      //  [56..63]     80             15.6 MHz
      //  Delay (bits 6:31) [units of TimingRef clk]
      vuint32_t offset;        //  Not implemented
      vuint32_t countAcquire;
      vuint32_t countEnable;
      vuint32_t countInhibit;
      vuint32_t countRead;
      vuint32_t countStart;
      vuint32_t countQueue;
      //
      vuint32_t cacheSel;
      vuint32_t cacheState;
      vuint32_t cacheAddr;

      vuint32_t msgDelay;
      vuint32_t headerCnt;

      vuint32_t rsvd_84[5];

      vuint32_t localId;
      vuint32_t upstreamId;
      vuint32_t dnstreamId[4];
      //
      //      vuint32_t status;
      //      vuint32_t statusCount[32];
    };
  };
};

#endif
