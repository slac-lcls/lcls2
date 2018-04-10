#ifndef Xpm_Module_hh
#define Xpm_Module_hh

#include "psdaq/cphw/Reg.hh"
#include "psdaq/cphw/Reg64.hh"
#include "psdaq/cphw/AmcPLL.hh"
#include "psdaq/cphw/AmcTiming.hh"
#include "psdaq/cphw/HsRepeater.hh"
#include "psdaq/cphw/RingBuffer.hh"

namespace Pds {
  namespace Xpm {

    class CoreCounts {
    public:
      void dump() const;
    public:
      uint64_t rxClkCount;
      uint64_t txClkCount;
      uint64_t rxRstCount;
      uint64_t crcErrCount;
      uint64_t rxDecErrCount;
      uint64_t rxDspErrCount;
      uint64_t bypassResetCount;
      uint64_t bypassDoneCount;
      uint64_t rxLinkUp;
      uint64_t fidCount;
      uint64_t sofCount;
      uint64_t eofCount;
    };

    class L0Stats {
    public:
      L0Stats();
      void dump() const;
    public:
      uint64_t l0Enabled;
      uint64_t l0Inhibited;
      uint64_t numl0;
      uint64_t numl0Inh;
      uint64_t numl0Acc;
      uint32_t linkInh[32];
      uint16_t rx0Errs;
    };

    class LinkStatus {
    public:
      LinkStatus();
    public:
      bool     txResetDone;
      bool     txReady;
      bool     rxResetDone;
      bool     rxReady;
      bool     isXpm;
      uint32_t rxRcvs;
      uint16_t rxErrs;
    };

    class XpmSequenceEngine;

    class Module {
    public:
      enum { NAmcs=2 };
      enum { NDSLinks=14 };
      enum { NPartitions=8 };
    public:
      static class Module* locate();
    public:
      Module();
      void init();
    public:
      Pds::Cphw::AmcTiming  _timing;
    private:
      uint32_t _reserved_AT[(0x09000000-sizeof(Module::_timing))>>2];
    public:
      Pds::Cphw::HsRepeater _hsRepeater[6];
    private:
      uint32_t _reserved_HR[(0x77000000-sizeof(Module::_hsRepeater))>>2];
    public:
      CoreCounts counts    () const;
      bool       l0Enabled () const;
      L0Stats    l0Stats   () const;
      unsigned   txLinkStat() const;
      unsigned   rxLinkStat() const;
    public:
      void clearLinks  ();
    public:
      LinkStatus linkStatus(unsigned) const;
      void       linkStatus(LinkStatus*) const;
      unsigned rxLinkErrs(unsigned) const;
    public:
      void resetL0     (bool);
      void resetL0     ();
      bool l0Reset     () const;
      void setL0Enabled(bool);
      bool getL0Enabled() const;
      void setL0Select_FixedRate(unsigned rate);
      void setL0Select_ACRate   (unsigned rate, unsigned tsmask);
      void setL0Select_Sequence (unsigned seq , unsigned bit);
      void setL0Select_Destn    (unsigned mode, unsigned mask);
      //      void setL0Select_EventCode(unsigned code);
      void lockL0Stats (bool);
      //    private:
    public:
      void setRingBChan(unsigned);
    public:
      void dumpPll     (unsigned) const;
      void dumpTiming  (unsigned) const;
      void pllBwSel    (unsigned, int);
      void pllFrqTbl   (unsigned, int);
      void pllFrqSel   (unsigned, int);
      void pllRateSel  (unsigned, int);
      void pllPhsInc   (unsigned);
      void pllPhsDec   (unsigned);
      void pllBypass   (unsigned, bool);
      void pllReset    (unsigned);
      int  pllBwSel  (unsigned) const;
      int  pllFrqTbl (unsigned) const;
      int  pllFrqSel (unsigned) const;
      int  pllRateSel(unsigned) const;
      bool pllBypass (unsigned) const;
      int  pllStatus0(unsigned) const;
      int  pllCount0 (unsigned) const;
      int  pllStatus1(unsigned) const;
      int  pllCount1 (unsigned) const;
      void pllSkew       (unsigned, int);
    public:
      // Indexing
      void setPartition(unsigned) const;
      void setLink     (unsigned) const;
      void setAmc      (unsigned) const;
      void setInhibit  (unsigned);
      void setTagStream(unsigned);
      unsigned getPartition() const;
      unsigned getLink     () const;
      unsigned getAmc      () const;
      unsigned getInhibit  () const;
      unsigned getTagStream() const;
    public:
      void     linkTxDelay(unsigned, unsigned);
      unsigned linkTxDelay(unsigned) const;
      void     linkPartition(unsigned, unsigned);
      unsigned linkPartition(unsigned) const;
      void     linkTrgSrc(unsigned, unsigned);
      unsigned linkTrgSrc(unsigned) const;
      void     linkLoopback(unsigned, bool);
      bool     linkLoopback(unsigned) const;
      void     txLinkReset (unsigned);
      void     rxLinkReset (unsigned);
      void     txLinkPllReset (unsigned);
      void     rxLinkPllReset (unsigned);
      void     rxLinkDump  (unsigned) const;
      void     linkEnable  (unsigned, bool);
      bool     linkEnable  (unsigned) const;
      bool     linkRxReady (unsigned) const;
      bool     linkTxReady (unsigned) const;
      bool     linkIsXpm   (unsigned) const;
      bool     linkRxErr   (unsigned) const;
    public:
      void     setL0Delay (unsigned, unsigned);
      unsigned getL0Delay (unsigned) const;
      void     setL1TrgClr(unsigned, unsigned);
      unsigned getL1TrgClr(unsigned) const;
      void     setL1TrgEnb(unsigned, unsigned);
      unsigned getL1TrgEnb(unsigned) const;
      void     setL1TrgSrc(unsigned, unsigned);
      unsigned getL1TrgSrc(unsigned) const;
      void     setL1TrgWord(unsigned, unsigned);
      unsigned getL1TrgWord(unsigned) const;
      void     setL1TrgWrite(unsigned, unsigned);
      unsigned getL1TrgWrite(unsigned) const;
    public:
      void     messagePayload(unsigned, unsigned);
      unsigned messagePayload(unsigned) const;
      void     messageHdr(unsigned, unsigned);  // inserts the message
      unsigned messageHdr(unsigned) const;
    public:
      void     inhibitInt(unsigned, unsigned);
      unsigned inhibitInt(unsigned) const;
      void     inhibitLim(unsigned, unsigned);
      unsigned inhibitLim(unsigned) const;
      void     inhibitEnb(unsigned, unsigned);
      unsigned inhibitEnb(unsigned) const;
    public:  // 0x80000000
      //  0x0000 - RO: physical link address
      Cphw::Reg   _paddr;
      //  0x0004 - RW: programming index
      //  [3:0]   partition     Partition number
      //  [9:4]   link          Link number
      //  [14:10] linkDebug     Link number for input to ring buffer
      //  [16]    amc           AMC selection
      //  [21:20] inhibit       Inhibit index
      //  [24]    tagStream     Enable tag FIFO streaming input
      Cphw::Reg   _index;
      //  0x0008 - RW: ds link configuration for link[index]
      //  [17:0]  txDelay       Transmit delay
      //  [18]    txPllReset    Transmit reset
      //  [19]    rxPllReset    Receive  reset
      //  [23:20] partition     Partition
      //  [27:24] trigsrc       Trigger source
      //  [28]    loopback      Loopback mode
      //  [29]    txReset       Transmit reset
      //  [30]    rxReset       Receive  reset
      //  [31]    enable        Enable
      Cphw::Reg  _dsLinkConfig;
      //  0x000C - RO: ds link status for link[index]
      //  [15:0]  rxErrCnts     Receive  error counts
      //  [16]    txResetDone   Transmit reset done
      //  [17]    txReady       Transmit ready
      //  [18]    rxResetDone   Receive  reset done
      //  [19]    rxReady       Receive  ready
      //  [20]    rxIsXpm       Remote side is XPM
      Cphw::Reg  _dsLinkStatus;
      //  [31:0]  rxRcvCnts
      Cphw::Reg  _dsLinkRcvs;

      Cphw::AmcPLL _amcPll;
      //  0x0014 - RW: L0 selection control for partition[index]
      //  [0]     reset
      //  [16]    enable
      //  [31]    enable counter update
      Cphw::Reg   _l0Control;
      //  0x0018 - RW: L0 selection criteria for partition[index]
      //  [15: 0]  rateSel      L0 rate selection
      //  [31:16]  destSel      L0 destination selection
      //
      //  [15:14]=00 (fixed rate), [3:0] fixed rate marker
      //  [15:14]=01 (ac rate),    [8:3] timeslot mask, [2:0] ac rate marker
      //  [15:14]=10 (sequence),   [13:8] sequencer, [3:0] sequencer bit
      //  [31]=1 any destination or match any of [14:0] mask of destinations
      Cphw::Reg   _l0Select;
      //  0x001c - RO: Clks enabled for partition[index]
      Cphw::Reg64 _l0Enabled;
      //  0x0024 - RO: Clks inhibited for partition[index]
      Cphw::Reg64 _l0Inhibited;
      //  0x002c - RO: Num L0s input for partition[index]
      Cphw::Reg64 _numl0;
      //  0x0034 - RO: Num L0s inhibited for partition[index]
      Cphw::Reg64 _numl0Inh;
      //  0x003c - RO: Num L0s accepted for partition[index]
      Cphw::Reg64 _numl0Acc;
      //  0x0044 - RO: Num L1s accepted for partition[index]
      Cphw::Reg64 _numl1Acc;
      //  0x004c - RW: L1 select config for partition[index]
      //  [0]     NL1Triggers clear  mask bits
      //  [16]    NL1Triggers enable mask bits
      Cphw::Reg   _l1config0;
      //  0x0050 - RW: L1 select config for partition[index]
      //  [3:0]   trigsrc       L1 trigger source link
      //  [12:4]  trigword      L1 trigger word
      //  [16]    trigwr        L1 trigger write mask
      Cphw::Reg   _l1config1;
      //  0x0054 - RW: Analysis tag reset for partition[index]
      //  [3:0]   reset
      Cphw::Reg   _analysisRst;
      //  0x0058 - RW: Analysis tag for partition[index]
      //  [31:0]  tag[3:0]
      Cphw::Reg   _analysisTag;
      //  0x005c - RW: Analysis push for partition[index]
      //  [3:0]   push
      Cphw::Reg   _analysisPush;
      //  0x0060 - RO: Analysis tag push counts for partition[index]
      Cphw::Reg   _analysisTagWr;
      //  0x0064 - RO: Analysis tag pull counts for partition[index]
      Cphw::Reg   _analysisTagRd;
      //  0x0068 - RW: Pipeline depth for partition[index]
      Cphw::Reg   _pipelineDepth;
      //  0x006c - RW: Message setup for partition[index]
      //  [14: 0]  Header
      //  [15]     Insert
      Cphw::Reg   _message;
      //  0x0070 - RW: Message payload for partition[index]
      Cphw::Reg   _messagePayload;
    private:
      uint32_t    _reserved_116[2];
    public:
      //  0x0080 - RW: Inhibit configurations for partition[index]
      //  [11:0]  interval      interval (929kHz ticks)
      //  [15:12] limit         max # accepts within interval
      //  [31]    enable        enable
      Cphw::Reg    _inhibitConfig[4];
      //  0x0090 - RO: Inhibit assertions by DS link for partition[index]
      Cphw::Reg    _inhibitCounts[32];
    public:
      //  0x0110 - RO: Monitor clock
      //  [28: 0]  Rate
      //  [29]     Slow
      //  [30]     Fast
      //  [31]     Lock
      Cphw::Reg _monClk[4];
    private:
      uint32_t    _reserved_168[(0x10000-288)>>2];
      //
      Cphw::RingBuffer _rxRing;  // 0x80010000
      uint32_t    _reserved_80020000[(0x10000-sizeof(_rxRing))>>2];
      
    public:
      XpmSequenceEngine& sequenceEngine();
    private:
      uint32_t    _reserved_engine[0x10000>>2];
    };
  };
};

#endif

