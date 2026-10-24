/**
 * @file
 * @brief Xpm::Module, the register map and control functions of an XPM (accessed through Cphw::Reg), and the statistics records it returns. Register bit layouts below are quoted from the comments in this header.
 */
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
  /** Namespace of the XPM register access and control code. */
  namespace Xpm {

    /** Snapshot of the counters of one timing receiver (Cphw::TimingRx), as built by Module::counts(). */
    class TimingCounts {
    public:
      /** Construct without initializing the counters. */
      TimingCounts() {}
      /** Copy the counters from the timing receiver (RxRecClks, TxRefClks, RxRstDone, CRCerrors, RxDecErrs, RxDspErrs, the two halves of BuffByCnts, bit 1 of CSR, Msgcounts, SOFcounts, EOFcounts); rxAlign is set to 0 and the alignment block is not read. */
      TimingCounts(const Cphw::TimingRx&,
                   const Cphw::GthRxAlign&);
    public:
      /** Print the counters (except rxAlign) in hex to stdout. */
      void dump() const;
    public:
      uint64_t rxClkCount;  ///< TimingRx RxRecClks.
      uint64_t txClkCount;  ///< TimingRx TxRefClks.
      uint64_t rxRstCount;  ///< TimingRx RxRstDone.
      uint64_t crcErrCount;  ///< TimingRx CRCerrors.
      uint64_t rxDecErrCount;  ///< TimingRx RxDecErrs.
      uint64_t rxDspErrCount;  ///< TimingRx RxDspErrs.
      uint64_t bypassResetCount;  ///< Upper 16 bits of TimingRx BuffByCnts.
      uint64_t bypassDoneCount;  ///< Lower 16 bits of TimingRx BuffByCnts.
      uint64_t rxLinkUp;  ///< Bit 1 of TimingRx CSR.
      uint64_t fidCount;  ///< TimingRx Msgcounts.
      uint64_t sofCount;  ///< TimingRx SOFcounts.
      uint64_t eofCount;  ///< TimingRx EOFcounts.
      uint64_t rxAlign;  ///< Always 0 (the alignment read is commented out).
    };

    /** Status of one AMC PLL, as returned by Module::pllStat(). */
    class PllStats {
    public:
      bool     lol;  ///< AmcPLL Status0() value (field named lol).
      bool     los;  ///< AmcPLL Status1() value (field named los).
      unsigned lolCount;  ///< AmcPLL Count0() value.
      unsigned losCount;  ///< AmcPLL Count1() value.
    };
      
    /** Counters of the two timing receivers, as returned by Module::counts(). */
    class CoreCounts {
    public:
      TimingCounts us;  ///< Counters of the us timing receiver (_usTiming).
      TimingCounts cu;  ///< Counters of the cu timing receiver (_cuTiming).
    };

    /** L0 trigger statistics of the selected partition, as sampled by Module::l0Stats(). */
    class L0Stats {
    public:
      /** Zero all counters and arrays (time is left unset). */
      L0Stats();
      /** Print the counters, the per-link inhibit time counts and rx0Errs to stdout. */
      void dump() const;
    public:
      uint64_t l0Enabled;  ///< Value of the _l0Enabled register (clocks enabled, per the header comment); master only.
      uint64_t l0Inhibited;  ///< Value of the _l0Inhibited register (clocks inhibited); master only.
      uint64_t numl0;  ///< Value of the _numl0 register (L0s input); master only.
      uint64_t numl0Inh;  ///< Value of the _numl0Inh register (L0s inhibited); master only.
      uint64_t numl0Acc;  ///< Value of the _numl0Acc register (L0s accepted); master only.
      uint32_t linkInhEv[32];  ///< Per-link _inhibitEvCounts values (inhibit assertions by link, per the header comment); master only.
      uint32_t linkInhTm[32];  ///< Per-link _inhibitTmCounts values.
      uint16_t rx0Errs;  ///< Receive error count of link 0.
      struct timespec time;  ///< CLOCK_REALTIME time at which the counters were sampled.
    };

    /** Status of one link, as returned by Module::linkStatus(). */
    class LinkStatus {
    public:
      /** Zero all flags and counters except remoteLinkId, which is left unset. */
      LinkStatus();
    public:
      bool     txResetDone;  ///< Link status bit 16 (transmit reset done, per the header comment).
      bool     txReady;  ///< Link status bit 17 (transmit ready).
      bool     rxResetDone;  ///< Link status bit 18 (receive reset done).
      bool     rxReady;  ///< Link status bit 19 (receive ready).
      bool     isXpm;  ///< Link status bit 20 (remote side is an XPM).
      uint32_t rxRcvs;  ///< Value of _dsLinkRcvs; set to all ones for link 16.
      uint16_t rxErrs;  ///< Link status bits 15-0 (receive error count); set to all ones for link 16.
      uint32_t remoteLinkId;  ///< Value of _remoteLinkId.
    };

    class XpmSequenceEngine;

    /** Register map of an XPM, laid out so that member addresses equal register offsets (see locate()), with functions that select a partition, link or AMC through the _index register and read or write fields of the selected registers. */
    class Module {
    public:
      /** Number of AMCs. */
      enum { NAmcs=2  /**< 2. */ };
      /** Number of downstream links. */
      enum { NDSLinks=14  /**< 14; init(), clearLinks(), txLinkStat() and rxLinkStat() loop over this many links. */ };
      /** Number of partitions. */
      enum { NPartitions=8  /**< 8. */ };
    public:
      /** Construct a Module with placement new at address 0 and return it, so that member addresses are register offsets for Cphw::Reg accesses. */
      static class Module* locate();
      /** Return the feature revision found by init(): 1 if the build stamp contains xtpg, else 0. */
      static unsigned      feature_rev();
    public:
      /** Does nothing (the call to init() is commented out). */
      Module();
      /** Set the feature revision from the build stamp, print the index register and the configuration and status of every link, route the crossbar outputs BP, RTM0 and RTM1 to FPGA and initialize HsRepeaters 0, 1, 3 and 4. */
      void init();
    public: //  AxiVersion @ 0
      /** AxiVersion block at offset 0 (per the comment); init() reads its build stamp. */
      Cphw::AxiVersion _version;
    private:
      uint32_t rsvd_version[(0x03000000-sizeof(_version))>>2];
    public: //  AxiSy56040 @ 0x03000000
      /** Crossbar (AxiSy56040 at 0x03000000, per the comment); init() and setCuInput() set its outputs. */
      Cphw::XBar       _xbar;
    private:
      uint32_t rsvd_xbar[(0x05000000-sizeof(_xbar))>>2];
    public: //  TimingRx   @ 0x08000000
      /** TimingRx block at 0x08000000 (per the comment); read by counts() as the us counters. */
      Cphw::TimingRx  _usTiming;
    private:
      uint32_t rsvd_us[(0x00400000-sizeof(_usTiming))>>2];
    public: //  TimingRx   @ 0x08400000
      /** TimingRx block at 0x08400000 (per the comment); read by counts() as the cu counters. */
      Cphw::TimingRx  _cuTiming;
    private:
      uint32_t rsvd_cu[(0x00400000-sizeof(_cuTiming))>>2];
    public: //  Generator  @ 0x08800000
      /** 64-bit timestamp register of the generator block at 0x08800000 (per the comment); written by setTimeStamp(). */
      Cphw::Reg64 _timestamp;
      Cphw::Reg64 _pulseId;  ///< 64-bit pulse ID register of the generator block.
      /** Delay register written by setCuDelay(); the comment gives 185.7 MHz units and a default of 800*200 clocks. */
      Cphw::Reg   _cuDelay;    // 185.7 MHz units (default 800*200 clocks)
      /** Register written by setCuBeamCode(); the comment calls it the beam-present event code (default 140). */
      Cphw::Reg   _cuBeamCode; // beam present eventcode (default 140)
      Cphw::Reg   _cuFiducialIntv;  ///< Register written by clearCuFiducialErr().
    private:
      uint32_t rsvd_gen[(0x00100000-28)>>2];
    public: //  MmcmPhaseLock @0x08900000,08a00000,08b00000
      /** Three MmcmPhaseLock blocks at 0x08900000, 0x08a00000 and 0x08b00000 (per the comment). */
      MmcmPhaseLock _mmcm[3];
    private:
      uint32_t _reserved_AT[(0x00400000)>>2];
    public: // HsRepeater  @ 0x09000000
      /** Six HsRepeater blocks from 0x09000000 (per the comment); init() initializes numbers 0, 1, 3 and 4. */
      Cphw::HsRepeater _hsRepeater[6];
    private:
      uint32_t _reserved_HR[(0x02000000-sizeof(Module::_hsRepeater))>>2];
    public: // GthRxAlign @ 0x0B000000
      /** GthRxAlign block at 0x0B000000 (per the comment); passed to TimingCounts but not read. */
      Cphw::GthRxAlign _usGthAlign;
    private:
      uint32_t _reservedUsGthAlign[(0x01000000-sizeof(_usGthAlign))>>2];
    public: // GthRxAlign @ 0x0C000000
      /** GthRxAlign block at 0x0C000000 (per the comment); passed to TimingCounts but not read. */
      Cphw::GthRxAlign _cuGthAlign;
    private:
      uint32_t _reservedCuGthAlign[(0x01000000-sizeof(_cuGthAlign))>>2];
      uint32_t _reservedToApp[(0x73000000)>>2];
    public:
      /** Return the counters of the us and cu timing receivers. */
      CoreCounts counts    () const;
      /** Return bit 16 (enable) of the _l0Control register of the selected partition. */
      bool       l0Enabled () const;
      /** Freeze the counter updates, sample the L0 statistics of the selected partition (the totals and per-link event inhibit counts only when the argument is true, the per-link inhibit time counts always) and the link 0 error count, then unfreeze. Note that the link index is changed while sampling. */
      L0Stats    l0Stats   (bool) const;
      /** Return a bit mask of the links (0 to NDSLinks-1) whose transmit-ready bit is set. */
      unsigned   txLinkStat() const;
      /** Return a bit mask of the links (0 to NDSLinks-1) whose receive-ready bit is set. */
      unsigned   rxLinkStat() const;
    public:
      /** Clear the enable bit (31) of the configuration of links 0 to NDSLinks-1. */
      void clearLinks  ();
    public:
      /** Select the link and return its status (see LinkStatus). */
      LinkStatus linkStatus(unsigned) const;
      /** Fill the array with the status of links 0 to 31. */
      void       linkStatus(LinkStatus*) const;
      /** Select the link and return its receive error count (status bits 15-0). */
      unsigned rxLinkErrs(unsigned) const;
    public:
      /** Set (true) or clear the reset bit (0) of _l0Control of the selected partition. */
      void resetL0     (bool);
      /** Pulse the reset bit of _l0Control: set it, wait 1 us, clear it. */
      void resetL0     ();
      /** Return the reset bit (0) of _l0Control. */
      bool l0Reset     () const;
      /** Set or clear the master bit (30) of _l0Control. */
      void master      (bool);
      /** Return the master bit (30) of _l0Control. */
      bool master      () const;
      /** Set or clear the enable bit (16) of _l0Control. */
      void setL0Enabled(bool);
      /** Return the enable bit (16) of _l0Control. */
      bool getL0Enabled() const;
      /** Set the rate selection of _l0Select to fixed-rate mode with marker rate (low 4 bits), keeping the destination selection. */
      void setL0Select_FixedRate(unsigned rate);
      /** Set the rate selection of _l0Select to AC-rate mode with timeslot mask tsmask (6 bits) and marker rate (3 bits), keeping the destination selection. */
      void setL0Select_ACRate   (unsigned rate, unsigned tsmask);
      /** Set the rate selection of _l0Select to sequence mode with sequencer seq (6 bits) and bit (4 bits), keeping the destination selection. */
      void setL0Select_Sequence (unsigned seq , unsigned bit);
      /** Set the destination selection (upper half) of _l0Select to mode in bit 15 and the low 4 bits of mask, keeping the rate selection. The header comment describes a 15-bit destination mask, but only 4 bits are kept. */
      void setL0Select_Destn    (unsigned mode, unsigned mask);
      //      void setL0Select_EventCode(unsigned code);
      /** Freeze (true) or resume the counter updates by clearing or setting bit 31 of _l0Control. */
      void lockL0Stats (bool);
      //    private:
      /** Select the partition of the lowest set bit of the mask, then write the mask to _groupL0Reset. */
      void groupL0Reset  (unsigned);
      /** Select the partition of the lowest set bit of the mask, then write the mask to _groupL0Enable. */
      void groupL0Enable (unsigned);
      /** Select the partition of the lowest set bit of the mask, then write the mask to _groupL0Disable. */
      void groupL0Disable(unsigned);
      /** Select the partition of the lowest set bit of the mask, then write the mask to _groupMsgInsert. */
      void groupMsgInsert(unsigned);
    public:
      /** Set the ring buffer link field (bits 13-10) of _index. */
      void setRingBChan(unsigned);
    public:
      /** Select the AMC and print its PLL configuration register and AmcPLL::dump(). */
      void dumpPll     (unsigned) const;
      /** Print that it is deprecated; does nothing else. */
      void dumpTiming  (unsigned) const;
      /** Set the file-static verbosity used by setL0Enabled() and l0Stats(). */
      void setVerbose  (unsigned);
      /** Write the current time to _timestamp: seconds since a reference from mktime() in the upper 32 bits and nanoseconds in the lower 32 bits. The reference struct tm has only tm_year (1995) set; its other fields are uninitialized. */
      void setTimeStamp();
      /** With feature revision 1 or more, route the crossbar outputs FPGA, RTM0 and RTM1 to input v (and print it); otherwise does nothing. */
      void setCuInput  (unsigned);
      /** With feature revision 1 or more, write v to _cuDelay. */
      void setCuDelay  (unsigned);
      /** With feature revision 1 or more, write v to _cuBeamCode. */
      void setCuBeamCode(unsigned);
      /** With feature revision 1 or more and v non-zero, write v to _cuFiducialIntv. */
      void clearCuFiducialErr(unsigned);
      /** Select AMC idx and call AmcPLL::BwSel(val). */
      void pllBwSel    (unsigned, int);
      /** Select AMC idx and call AmcPLL::FrqTbl(val). */
      void pllFrqTbl   (unsigned, int);
      /** Select AMC idx and call AmcPLL::FrqSel(val). */
      void pllFrqSel   (unsigned, int);
      /** Select AMC idx and call AmcPLL::RateSel(val). */
      void pllRateSel  (unsigned, int);
      /** Select AMC idx and call AmcPLL::PhsInc(). */
      void pllPhsInc   (unsigned);
      /** Select AMC idx and call AmcPLL::PhsDec(). */
      void pllPhsDec   (unsigned);
      /** Select AMC idx and call AmcPLL::Bypass(v). */
      void pllBypass   (unsigned, bool);
      /** Select AMC idx and call AmcPLL::Reset(). */
      void pllReset    (unsigned);
      /** Select AMC idx and return AmcPLL::BwSel(). */
      int  pllBwSel  (unsigned) const;
      /** Select AMC idx and return AmcPLL::FrqTbl(). */
      int  pllFrqTbl (unsigned) const;
      /** Select AMC idx and return AmcPLL::FrqSel(). */
      int  pllFrqSel (unsigned) const;
      /** Select AMC idx and return AmcPLL::RateSel(). */
      int  pllRateSel(unsigned) const;
      /** Select AMC idx and return AmcPLL::Bypass(). */
      bool pllBypass (unsigned) const;
      /** Select AMC idx and return AmcPLL::Status0(). */
      int  pllStatus0(unsigned) const;
      /** Select AMC idx and return AmcPLL::Count0(). */
      int  pllCount0 (unsigned) const;
      /** Select AMC idx and return AmcPLL::Status1(). */
      int  pllStatus1(unsigned) const;
      /** Select AMC idx and return AmcPLL::Count1(). */
      int  pllCount1 (unsigned) const;
      /** Select AMC idx and call AmcPLL::Skew(val). */
      void pllSkew       (unsigned, int);
      /** Select AMC idx and return its PLL status (Status0, Status1, Count0, Count1). */
      PllStats pllStat(unsigned) const;
    public:
      // Indexing
      /** Set the partition field (bits 3-0) of _index; const although it writes the register. */
      void setPartition(unsigned) const;
      /** Set the link field (bits 9-4) of _index; const although it writes the register. */
      void setLink     (unsigned) const;
      /** Set the AMC bit (16) of _index; const although it writes the register. */
      void setAmc      (unsigned) const;
      /** Set the inhibit field (bits 21-20) of _index. */
      void setInhibit  (unsigned);
      /** Set the tag stream bit (24) of _index. */
      void setTagStream(unsigned);
      /** Return the partition field (bits 3-0) of _index. */
      unsigned getPartition() const;
      /** Return the link field (bits 9-4) of _index. */
      unsigned getLink     () const;
      /** Return the AMC bit (16) of _index. */
      unsigned getAmc      () const;
      /** Return the inhibit field (bits 21-20) of _index. */
      unsigned getInhibit  () const;
      /** Return the tag stream bit (24) of _index. */
      unsigned getTagStream() const;
    public:
      /** Select the link and set its receive timeout field (bits 17-9 of _dsLinkConfig). */
      void     linkRxTimeOut(unsigned, unsigned);
      /** Select the link and return its receive timeout field (bits 17-9). */
      unsigned linkRxTimeOut(unsigned) const;
      /** Select the link and set its group mask (bits 7-0 of _dsLinkConfig). */
      void     linkGroupMask(unsigned, unsigned);
      /** Select the link and return its group mask (bits 7-0). */
      unsigned linkGroupMask(unsigned) const;
      /** Select the link and set its trigger source field (bits 27-24 of _dsLinkConfig). */
      void     linkTrgSrc(unsigned, unsigned);
      /** Select the link and return its trigger source field (bits 27-24). */
      unsigned linkTrgSrc(unsigned) const;
      /** Select the link and set or clear its loopback bit (28 of _dsLinkConfig). */
      void     linkLoopback(unsigned, bool);
      /** Select the link and return its loopback bit (28). */
      bool     linkLoopback(unsigned) const;
      /** Select the link and pulse its transmit reset bit (29 of _dsLinkConfig) for 10 us. */
      void     txLinkReset (unsigned);
      /** Select the link and pulse its receive reset bit (30 of _dsLinkConfig) for 10 us. */
      void     rxLinkReset (unsigned);
      /** Select the link and pulse its transmit PLL reset bit (18 of _dsLinkConfig) for 10 us. */
      void     txLinkPllReset (unsigned);
      /** Select the link and pulse its receive PLL reset bit (19 of _dsLinkConfig) for 10 us. */
      void     rxLinkPllReset (unsigned);
      /** Point the receive ring buffer at the link, clear it, capture for 100 us and print the captured words masked to 20 bits (Cphw::RingBuffer::dump(20)). */
      void     rxLinkDump  (unsigned) const;
      /** Select the link and set or clear its enable bit (31 of _dsLinkConfig). */
      void     linkEnable  (unsigned, bool);
      /** Select the link and return its enable bit (31). */
      bool     linkEnable  (unsigned) const;
      /** Select the link and return its receive-ready status bit (19 of _dsLinkStatus). */
      bool     linkRxReady (unsigned) const;
      /** Select the link and return its transmit-ready status bit (17). */
      bool     linkTxReady (unsigned) const;
      /** Select the link and return its remote-is-XPM status bit (20). */
      bool     linkIsXpm   (unsigned) const;
      /** Select the link and return true if its receive error count (status bits 15-0) is non-zero. */
      bool     linkRxErr   (unsigned) const;
    public:
      /** Write the delay v to the upper 16 bits of _pipelineDepth and the low 16 bits of v*200 to its lower half. */
      void     setL0Delay (unsigned);
      /** Return the upper 16 bits of _pipelineDepth. */
      unsigned getL0Delay () const;
      /** Set bit 0 of _l1config0 to v. */
      void     setL1TrgClr(unsigned);
      /** Return bit 0 of _l1config0. */
      unsigned getL1TrgClr() const;
      /** Set bit 16 of _l1config0 to v. */
      void     setL1TrgEnb(unsigned);
      /** Return bit 16 of _l1config0. */
      unsigned getL1TrgEnb() const;
      /** Set bits 3-0 of _l1config1 (L1 trigger source link, per the header comment). */
      void     setL1TrgSrc(unsigned);
      /** Return bits 3-0 of _l1config1. */
      unsigned getL1TrgSrc() const;
      /** Set bits 12-4 of _l1config1 (L1 trigger word). */
      void     setL1TrgWord(unsigned);
      /** Return bits 12-4 of _l1config1. */
      unsigned getL1TrgWord() const;
      /** Set bit 16 of _l1config1 (L1 trigger write mask). */
      void     setL1TrgWrite(unsigned);
      /** Return bit 16 of _l1config1. */
      unsigned getL1TrgWrite() const;
    public:
      /** Write v to _messagePayload of the selected partition. */
      void     messagePayload(unsigned);
      /** Return _messagePayload of the selected partition. */
      unsigned messagePayload() const;
      /** Set bits 7-0 of _message to v (the header comment gives the header as bits 14-0). */
      void     messageHdr(unsigned);
      /** Return bits 15-0 of _message. */
      unsigned messageHdr() const;
      /** Set the insert bit (15) of _message. */
      void     messageInsert();  // inserts the message
    public:
      /** Set the interval field of inhibit configuration inh (bits 11-0 of _inhibitConfig) to v - 1. */
      void     inhibitInt(unsigned, unsigned);
      /** Return the interval field of inhibit configuration inh plus 1. */
      unsigned inhibitInt(unsigned) const;
      /** Set the limit field of inhibit configuration inh (bits 15-12) to v - 1. */
      void     inhibitLim(unsigned, unsigned);
      /** Return the limit field of inhibit configuration inh plus 1. */
      unsigned inhibitLim(unsigned) const;
      /** Set the enable bit (31) of inhibit configuration inh to v. */
      void     inhibitEnb(unsigned, unsigned);
      /** Return the enable bit (31) of inhibit configuration inh. */
      unsigned inhibitEnb(unsigned) const;
    public:  // 0x80000000
      //  0x0000 - RW: physical link address (R: received address, W: transmit address)
      /** Register 0x0000: physical link address (read: received address, write: transmit address), per the header comment. */
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
      /** Register 0x0004: programming index selecting the partition [3:0], link [9:4], ring buffer link [14:10], AMC [16], inhibit [21:20], tag streaming [24] and the us/cu receive enables [25]/[26] (header comment). */
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
      /** Register 0x0008: configuration of the selected link: group mask [7:0], receive timeout [17:9], PLL resets [18]/[19], trigger source [27:24], loopback [28], resets [29]/[30], enable [31] (header comment). */
      Cphw::Reg  _dsLinkConfig;
      //  0x000C - RO: ds link status for link[index]
      //  [15:0]  rxErrCnts     Receive  error counts
      //  [16]    txResetDone   Transmit reset done
      //  [17]    txReady       Transmit ready
      //  [18]    rxResetDone   Receive  reset done
      //  [19]    rxReady       Receive  ready
      //  [20]    rxIsXpm       Remote side is XPM
      /** Register 0x000C (read-only): status of the selected link: receive error count [15:0], reset-done and ready bits [16]-[19], remote-is-XPM [20] (header comment). */
      Cphw::Reg  _dsLinkStatus;
      //  [31:0]  rxRcvCnts
      /** Receive count register of the selected link (bits 31-0, per the header comment). */
      Cphw::Reg  _dsLinkRcvs;
      //  0x0014 - 
      /** AMC PLL control block at 0x0014; the AMC is selected by _index. */
      Cphw::AmcPLL _amcPll;
      //  0x0018 - RW: L0 selection control for partition[index]
      //  [0]     reset
      //  [16]    enable
      //  [30]    master
      //  [31]    enable counter update
      /** Register 0x0018: L0 control of the selected partition: reset [0], enable [16], master [30], enable counter update [31] (header comment). */
      Cphw::Reg   _l0Control;
      //  0x001c - RW: L0 selection criteria for partition[index]
      //  [15: 0]  rateSel      L0 rate selection
      //  [31:16]  destSel      L0 destination selection
      //
      //  [15:14]=00 (fixed rate), [3:0] fixed rate marker
      //  [15:14]=01 (ac rate),    [8:3] timeslot mask, [2:0] ac rate marker
      //  [15:14]=10 (sequence),   [13:8] sequencer, [3:0] sequencer bit
      //  [31]=1 any destination or match any of [14:0] mask of destinations
      /** Register 0x001c: L0 selection of the selected partition: rate selection [15:0] and destination selection [31:16]; the rate modes are given by bits [15:14] (header comment). */
      Cphw::Reg   _l0Select;
      //  0x0020 - RO: Clks enabled for partition[index]
      /** Register 0x0020 (read-only, 64-bit): clocks enabled for the selected partition (header comment). */
      Cphw::Reg64 _l0Enabled;
      //  0x0028 - RO: Clks inhibited for partition[index]
      /** Register 0x0028 (read-only, 64-bit): clocks inhibited for the selected partition. */
      Cphw::Reg64 _l0Inhibited;
      //  0x0030 - RO: Num L0s input for partition[index]
      /** Register 0x0030 (read-only, 64-bit): number of L0s input for the selected partition. */
      Cphw::Reg64 _numl0;
      //  0x0038 - RO: Num L0s inhibited for partition[index]
      /** Register 0x0038 (read-only, 64-bit): number of L0s inhibited. */
      Cphw::Reg64 _numl0Inh;
      //  0x0040 - RO: Num L0s accepted for partition[index]
      /** Register 0x0040 (read-only, 64-bit): number of L0s accepted. */
      Cphw::Reg64 _numl0Acc;
      //  0x0048 - RO: Num L1s accepted for partition[index]
      /** Register 0x0048 (read-only, 64-bit): number of L1s accepted. */
      Cphw::Reg64 _numl1Acc;
      //  0x0050 - RW: L1 select config for partition[index]
      //  [0]     NL1Triggers clear  mask bits
      //  [16]    NL1Triggers enable mask bits
      /** Register 0x0050: L1 select configuration: trigger clear mask [0], trigger enable mask [16] (header comment). */
      Cphw::Reg   _l1config0;
      //  0x0054 - RW: L1 select config for partition[index]
      //  [3:0]   trigsrc       L1 trigger source link
      //  [12:4]  trigword      L1 trigger word
      //  [16]    trigwr        L1 trigger write mask
      /** Register 0x0054: L1 trigger source link [3:0], trigger word [12:4], write mask [16] (header comment). */
      Cphw::Reg   _l1config1;
      //  0x0058 - RW: Analysis tag reset for partition[index]
      //  [3:0]   reset
      /** Register 0x0058: analysis tag reset [3:0] for the selected partition (header comment). */
      Cphw::Reg   _analysisRst;
      //  0x005c - RW: Analysis tag for partition[index]
      //  [31:0]  tag[3:0]
      /** Register 0x005c: analysis tag for the selected partition (header comment). */
      Cphw::Reg   _analysisTag;
      //  0x0060 - RW: Analysis push for partition[index]
      //  [3:0]   push
      /** Register 0x0060: analysis push [3:0] for the selected partition (header comment). */
      Cphw::Reg   _analysisPush;
      //  0x0064 - RO: Analysis tag push counts for partition[index]
      /** Register 0x0064 (read-only): analysis tag push counts (header comment). */
      Cphw::Reg   _analysisTagWr;
      //  0x0068 - RO: Analysis tag pull counts for partition[index]
      /** Register 0x0068 (read-only): analysis tag pull counts (header comment). */
      Cphw::Reg   _analysisTagRd;
      //  0x006c - RW: Pipeline depth for partition[index]
      /** Register 0x006c: pipeline depth for the selected partition; written by setL0Delay(). */
      Cphw::Reg   _pipelineDepth;
      //  0x0070 - RW: Message setup for partition[index]
      //  [14: 0]  Header
      //  [15]     Insert
      /** Register 0x0070: message setup: header [14:0], insert [15] (header comment). */
      Cphw::Reg   _message;
      //  0x0074 - RW: Message payload for partition[index]
      /** Register 0x0074: message payload for the selected partition. */
      Cphw::Reg   _messagePayload;
      //  0x0078 - RO: Remote Link ID
      /** Register 0x0078 (read-only): remote link ID, copied into LinkStatus. */
      Cphw::Reg   _remoteLinkId;
    private:
      uint32_t    _reserved_120[1];
    public:
      //  0x0080 - RW: Inhibit configurations for partition[index]
      //  [11:0]  interval      interval (929kHz ticks)
      //  [15:12] limit         max # accepts within interval
      //  [31]    enable        enable
      /** Registers 0x0080: four inhibit configurations: interval [11:0], limit [15:12], enable [31] (header comment). */
      Cphw::Reg    _inhibitConfig[4];
      //  0x0090 - RO: Inhibit assertions by DS link for partition[index]
      /** Registers 0x0090 (read-only): inhibit assertions by downstream link (header comment); read by l0Stats(). */
      Cphw::Reg    _inhibitEvCounts[32];
    public:
      //  0x0110 - RO: Monitor clock
      //  [28: 0]  Rate
      //  [29]     Slow
      //  [30]     Fast
      //  [31]     Lock
      /** Registers 0x0110 (read-only): four monitor clocks: rate [28:0], slow [29], fast [30], lock [31] (header comment). */
      Cphw::Reg _monClk[4];
    public:
      Cphw::Reg    _inhibitTmCounts[32];  ///< Per-link counters following _monClk, read by l0Stats() into linkInhTm; not described in the header.
      uint32_t     _reserved_416[24];  ///< Unused padding (24 words) up to register 0x0200.
      //  0x0200 - WO: L0Reset
      /** Register 0x0200 (write-only): written with a group mask by groupL0Reset(). */
      Cphw::Reg    _groupL0Reset;
      //  0x0204 - WO: L0Enable
      /** Register 0x0204 (write-only): written with a group mask by groupL0Enable(). */
      Cphw::Reg    _groupL0Enable;
      //  0x0208 - WO: L0Disable
      /** Register 0x0208 (write-only): written with a group mask by groupL0Disable(). */
      Cphw::Reg    _groupL0Disable;
      //  0x020c - WO: MsgInsert
      /** Register 0x020c (write-only): written with a group mask by groupMsgInsert(). */
      Cphw::Reg    _groupMsgInsert;
    private:
      uint32_t    _reserved_528[(0x10000-0x210)>>2];
      //
      Cphw::RingBuffer _rxRing;  // 0x80010000
      uint32_t    _reserved_80020000[(0x10000-sizeof(_rxRing))>>2];
      
    public:
      /** Return a new XpmSequenceEngine for the register space after the receive ring buffer; a new object is allocated on every call and not freed here. */
      XpmSequenceEngine& sequenceEngine();
    private:
      uint32_t    _reserved_engine[0x10000>>2];
      uint32_t    _reserved_gthTSim[0x10000>>2];
    public:
      MmcmPhaseLock _mmcm_amc;  ///< MmcmPhaseLock block placed after the sequence engine and gthTSim reserved spaces.
    };
  };
};

#endif

