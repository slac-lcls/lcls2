#ifndef HSD_ChipAdcReg_hh
#define HSD_ChipAdcReg_hh

#include "psdaq/mmhw/Reg.hh"

#include "Globals.hh"

namespace Pds {
  namespace HSD {

    class ChipAdcReg {
    public:
      void init();
      void setChannels(unsigned);
      void setupDaq (unsigned partition);
      void start();
      void stop ();
      void resetCounts();
      void resetClock (bool);
      void resetDma   ();
      void resetFb    ();
      void resetFbPLL ();
      void setLocalId (unsigned v); // deprecated
      void dump() const;
    public:
      Mmhw::Reg irqEnable;
      Mmhw::Reg irqStatus;
      //      Mmhw::Reg partitionAddr; // deprecated
      uint32_t  rsvd_08;
      Mmhw::Reg dmaFullThr;
      Mmhw::Reg csr; 
      //   CSR
      // [ 0:0]  count reset
      // [ 1:1]  dma size histogram enable (unused)
      // [ 2:2]  dma test pattern enable
      // [ 3:3]  adc sync reset
      // [ 4:4]  dma reset
      // [ 5:5]  fb phy reset
      // [ 6:6]  fb pll reset
      // [ 8:15] trigger bit mask shift
      // [31:31] acqEnable
      //      Mmhw::Reg acqSelect; [ deprecated ]
      uint32_t  rsvd_20;
      //   AcqSelect
      // [12: 0]  rateSel  :   [7:0] eventCode
      // [31:13]  destSel
      Mmhw::Reg control;
      //   Control
      // [7:0] channel enable mask
      // [8:8] interleave
      // [19:16] readout group [ deprecated ]
      // [20]  inhibit
      Mmhw::Reg samples;       //  Must be a multiple of 16 [v1 only]
      Mmhw::Reg prescale;      //  Sample prescaler [v1 only]  [0x020]
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
      Mmhw::Reg offset;        // [0x24] Not implemented
      Mmhw::Reg countEnable;   // [0x28]
      Mmhw::Reg countAcquire;  // [0x2c]
      Mmhw::Reg countInhibit;  // [0x30]
      Mmhw::Reg countRead;     // [0x34]
      Mmhw::Reg countStart;    // [0x38]
      //      Mmhw::Reg countQueue;    // [ deprecated ]
      uint32_t  rsvd_60[1];
      //
      Mmhw::Reg cacheSel;
      Mmhw::Reg cacheState;
      Mmhw::Reg cacheAddr;

      // Mmhw::Reg msgDelay;       // [ deprecated ]
      // Mmhw::Reg headerCnt;      // [ deprecated ]
      // Mmhw::Reg headerFifo;     // [ deprecated ]

      // Mmhw::Reg rsvd_88[4];

      // Mmhw::Reg localId;        // [ deprecated ]
      // Mmhw::Reg upstreamId;     // [ deprecated ]
      // Mmhw::Reg dnstreamId[4];  // [ deprecated ]

      // Mmhw::Reg fullToTrig;     // [ deprecated ]
      //
      Mmhw::Reg buildStatus;
      //      Mmhw::Reg statusCount[32];
    };
  };
};

#endif
