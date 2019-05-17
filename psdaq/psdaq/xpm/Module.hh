#ifndef Xpm_Module_hh
#define Xpm_Module_hh

#include "psdaq/cphw/Reg.hh"
#include "psdaq/cphw/Reg64.hh"
#include "psdaq/cphw/AmcPLL.hh"
#include "psdaq/cphw/AxiVersion.hh"
#include "psdaq/cphw/GthRxAlign.hh"
#include "psdaq/cphw/TimingRx.hh"
#include "psdaq/cphw/HsRepeater.hh"
#include "psdaq/cphw/RingBuffer.hh"
#include "psdaq/cphw/XBar.hh"
#include "psdaq/xpm/MmcmPhaseLock.hh"

namespace Pds {
  namespace Xpm {

    class TimingCounts {
    public:
      TimingCounts() {}
      TimingCounts(const Cphw::TimingRx&,
                   const Cphw::GthRxAlign&);
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
      uint64_t rxAlign;
    };

    class PllStats {
    public:
      bool     lol;
      bool     los;
      unsigned lolCount;
      unsigned losCount;
    };
      
    class CoreCounts {
    public:
      TimingCounts us;
      TimingCounts cu;
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
      uint32_t linkInhEv[32];
      uint32_t linkInhTm[32];
      uint16_t rx0Errs;
      struct timespec time;
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
      uint32_t remoteLinkId;
    };

    class XpmSequenceEngine;

    class Module {
    public:
      enum { NAmcs=2 };
      enum { NDSLinks=14 };
      enum { NPartitions=8 };
    public:
      static class Module* locate();
      static unsigned      feature_rev();
    public:
      Module();
      void init();
    public: //  AxiVersion @ 0
      Cphw::AxiVersion _version;
    private:
      uint32_t rsvd_version[(0x03000000-sizeof(_version))>>2];
    public: //  AxiSy56040 @ 0x03000000
      Cphw::XBar       _xbar;
    private:
      uint32_t rsvd_xbar[(0x05000000-sizeof(_xbar))>>2];
    public: //  TimingRx   @ 0x08000000
      Cphw::TimingRx  _usTiming;
    private:
      uint32_t rsvd_us[(0x00400000-sizeof(_usTiming))>>2];
    public: //  TimingRx   @ 0x08400000
      Cphw::TimingRx  _cuTiming;
    private:
      uint32_t rsvd_cu[(0x00400000-sizeof(_cuTiming))>>2];
    public: //  Generator  @ 0x08800000
      Cphw::Reg64 _timestamp;
      Cphw::Reg64 _pulseId;
      Cphw::Reg   _cuDelay;    // 185.7 MHz units (default 800*200 clocks)
      Cphw::Reg   _cuBeamCode; // beam present eventcode (default 140)
      Cphw::Reg   _cuFiducialIntv; 
    private:
      uint32_t rsvd_gen[(0x00100000-28)>>2];
    public: //  MmcmPhaseLock @0x08900000,08a00000,08b00000
      MmcmPhaseLock _mmcm[3];
    private:
      uint32_t _reserved_AT[(0x00400000)>>2];
    public: // HsRepeater  @ 0x09000000
      Cphw::HsRepeater _hsRepeater[6];
    private:
      uint32_t _reserved_HR[(0x02000000-sizeof(Module::_hsRepeater))>>2];
    public: // GthRxAlign @ 0x0B000000
      Cphw::GthRxAlign _usGthAlign;
    private:
      uint32_t _reservedUsGthAlign[(0x01000000-sizeof(_usGthAlign))>>2];
    public: // GthRxAlign @ 0x0C000000
      Cphw::GthRxAlign _cuGthAlign;
    private:
      uint32_t _reservedCuGthAlign[(0x01000000-sizeof(_cuGthAlign))>>2];
      uint32_t _reservedToApp[(0x73000000)>>2];
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
      void master      (bool);
      bool master      () const;
      void setL0Enabled(bool);
      bool getL0Enabled() const;
      void setL0Select_FixedRate(unsigned rate);
      void setL0Select_ACRate   (unsigned rate, unsigned tsmask);
      void setL0Select_Sequence (unsigned seq , unsigned bit);
      void setL0Select_Destn    (unsigned mode, unsigned mask);
      //      void setL0Select_EventCode(unsigned code);
      void lockL0Stats (bool);
      //    private:
      void groupL0Reset  (unsigned);
      void groupL0Enable (unsigned);
      void groupL0Disable(unsigned);
      void groupMsgInsert(unsigned);
    public:
      void setRingBChan(unsigned);
    public:
      void dumpPll     (unsigned) const;
      void dumpTiming  (unsigned) const;
      void setVerbose  (unsigned);
      void setTimeStamp();
      void setCuInput  (unsigned);
      void setCuDelay  (unsigned);
      void setCuBeamCode(unsigned);
      void clearCuFiducialErr(unsigned);
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
      PllStats pllStat(unsigned) const;
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
      void     linkRxTimeOut(unsigned, unsigned);
      unsigned linkRxTimeOut(unsigned) const;
      void     linkGroupMask(unsigned, unsigned);
      unsigned linkGroupMask(unsigned) const;
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
      void     setL0Delay (unsigned);
      unsigned getL0Delay () const;
      void     setL1TrgClr(unsigned);
      unsigned getL1TrgClr() const;
      void     setL1TrgEnb(unsigned);
      unsigned getL1TrgEnb() const;
      void     setL1TrgSrc(unsigned);
      unsigned getL1TrgSrc() const;
      void     setL1TrgWord(unsigned);
      unsigned getL1TrgWord() const;
      void     setL1TrgWrite(unsigned);
      unsigned getL1TrgWrite() const;
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
      //  0x0000 - RW: physical link address (R: received address, W: transmit address)
      Cphw::Reg   _paddr;
      //  0x0004 - RW: programming index
      //  [3:0]   partition     Partition number
      //  [9:4]   link          Link number
      //  [14:10] linkDebug     Link number for input to ring buffer
      //  [16]    amc           AMC selection
      //  [21:20] inhibit       Inhibit index
      //  [24]    tagStream     Enable tag FIFO streaming input
      //  [25]    usRxEnable
      //  [26]    cuRxEnable
      Cphw::Reg   _index;
      //  0x0008 - RW: ds link configuration for link[index]
      //  [7:0]   groupMask     Full mask of groups
      //  [17:9]  rxTimeOut     Receive timeout
      //  [18]    txPllReset    Transmit reset
      //  [19]    rxPllReset    Receive  reset
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
      //  0x0014 - 
      Cphw::AmcPLL _amcPll;
      //  0x0018 - RW: L0 selection control for partition[index]
      //  [0]     reset
      //  [16]    enable
      //  [30]    master
      //  [31]    enable counter update
      Cphw::Reg   _l0Control;
      //  0x001c - RW: L0 selection criteria for partition[index]
      //  [15: 0]  rateSel      L0 rate selection
      //  [31:16]  destSel      L0 destination selection
      //
      //  [15:14]=00 (fixed rate), [3:0] fixed rate marker
      //  [15:14]=01 (ac rate),    [8:3] timeslot mask, [2:0] ac rate marker
      //  [15:14]=10 (sequence),   [13:8] sequencer, [3:0] sequencer bit
      //  [31]=1 any destination or match any of [14:0] mask of destinations
      Cphw::Reg   _l0Select;
      //  0x0020 - RO: Clks enabled for partition[index]
      Cphw::Reg64 _l0Enabled;
      //  0x0028 - RO: Clks inhibited for partition[index]
      Cphw::Reg64 _l0Inhibited;
      //  0x0030 - RO: Num L0s input for partition[index]
      Cphw::Reg64 _numl0;
      //  0x0038 - RO: Num L0s inhibited for partition[index]
      Cphw::Reg64 _numl0Inh;
      //  0x0040 - RO: Num L0s accepted for partition[index]
      Cphw::Reg64 _numl0Acc;
      //  0x0048 - RO: Num L1s accepted for partition[index]
      Cphw::Reg64 _numl1Acc;
      //  0x0050 - RW: L1 select config for partition[index]
      //  [0]     NL1Triggers clear  mask bits
      //  [16]    NL1Triggers enable mask bits
      Cphw::Reg   _l1config0;
      //  0x0054 - RW: L1 select config for partition[index]
      //  [3:0]   trigsrc       L1 trigger source link
      //  [12:4]  trigword      L1 trigger word
      //  [16]    trigwr        L1 trigger write mask
      Cphw::Reg   _l1config1;
      //  0x0058 - RW: Analysis tag reset for partition[index]
      //  [3:0]   reset
      Cphw::Reg   _analysisRst;
      //  0x005c - RW: Analysis tag for partition[index]
      //  [31:0]  tag[3:0]
      Cphw::Reg   _analysisTag;
      //  0x0060 - RW: Analysis push for partition[index]
      //  [3:0]   push
      Cphw::Reg   _analysisPush;
      //  0x0064 - RO: Analysis tag push counts for partition[index]
      Cphw::Reg   _analysisTagWr;
      //  0x0068 - RO: Analysis tag pull counts for partition[index]
      Cphw::Reg   _analysisTagRd;
      //  0x006c - RW: Pipeline depth for partition[index]
      Cphw::Reg   _pipelineDepth;
      //  0x0070 - RW: Message setup for partition[index]
      //  [14: 0]  Header
      //  [15]     Insert
      Cphw::Reg   _message;
      //  0x0074 - RW: Message payload for partition[index]
      Cphw::Reg   _messagePayload;
      //  0x0078 - RO: Remote Link ID
      Cphw::Reg   _remoteLinkId;
    private:
      uint32_t    _reserved_120[1];
    public:
      //  0x0080 - RW: Inhibit configurations for partition[index]
      //  [11:0]  interval      interval (929kHz ticks)
      //  [15:12] limit         max # accepts within interval
      //  [31]    enable        enable
      Cphw::Reg    _inhibitConfig[4];
      //  0x0090 - RO: Inhibit assertions by DS link for partition[index]
      Cphw::Reg    _inhibitEvCounts[32];
    public:
      //  0x0110 - RO: Monitor clock
      //  [28: 0]  Rate
      //  [29]     Slow
      //  [30]     Fast
      //  [31]     Lock
      Cphw::Reg _monClk[4];
    public:
      Cphw::Reg    _inhibitTmCounts[32];
      uint32_t     _reserved_416[24];
      //  0x0200 - WO: L0Reset
      Cphw::Reg    _groupL0Reset;
      //  0x0204 - WO: L0Enable
      Cphw::Reg    _groupL0Enable;
      //  0x0208 - WO: L0Disable
      Cphw::Reg    _groupL0Disable;
      //  0x020c - WO: MsgInsert
      Cphw::Reg    _groupMsgInsert;
    private:
      uint32_t    _reserved_528[(0x10000-0x210)>>2];
      //
      Cphw::RingBuffer _rxRing;  // 0x80010000
      uint32_t    _reserved_80020000[(0x10000-sizeof(_rxRing))>>2];
      
    public:
      XpmSequenceEngine& sequenceEngine();
    private:
      uint32_t    _reserved_engine[0x10000>>2];
      uint32_t    _reserved_gthTSim[0x10000>>2];
    public:
      MmcmPhaseLock _mmcm_amc;
    };
  };
};

#endif

