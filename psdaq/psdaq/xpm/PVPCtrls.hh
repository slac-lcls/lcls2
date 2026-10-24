/**
 * @file
 * @brief PVPCtrls, the EPICS control PVs of one XPM partition (used by xpmPVs).
 */
#ifndef Xpm_PVPCtrls_hh
#define Xpm_PVPCtrls_hh

#include <string>
#include <vector>

namespace Pds_Epics { class PVBase; }

namespace Pds {

  class Semaphore;

  namespace Xpm {

    class Module;
    class PVPStats;

    /** EPICS control PVs of one partition of an XPM; their callbacks program the L0 select, L1 trigger, analysis tag, message and inhibit registers of that partition in the Module. Apart from the XPM PV, a PV callback acts only while enabled(), that is while the value of the XPM PV equals this shelf. */
    class PVPCtrls
    {
    public:
      /** Store the module, semaphore, statistics object, shelf and partition; starts disabled and without PVs. The statistics object is stored but not used in PVPCtrls.cc. */
      PVPCtrls(Module&,
               Semaphore&,
               PVPStats&,
               unsigned shelf,
               unsigned partition);
      /** Does nothing; the PVs created by allocate() are not deleted. */
      ~PVPCtrls();
    public:
      /** Delete the PVs of an earlier call and create one PV per control, named title, a colon and the control name (XPM, Run, L0Select, MsgConfig and others). InhInterval, InhLimit and InhEnable get four PVs each, with suffixes 0 to 3. */
      void allocate(const std::string& title);
      /** Enable this object if shelf equals its shelf, else disable it. Under the semaphore, reset L0, set the Module master bit to the result and, if disabled, clear the L0 enable; if enabled, then call updated() on every PV except the first eight (XPM, Run, the message PVs and others). */
      void enable(unsigned shelf);
      /** Declared but not defined in PVPCtrls.cc. */
      void update();
      /** Return the result of the last enable() call (false before any call). */
      bool enabled() const;
      /** Select this partition in the Module (Module::setPartition()); the callers in PVPCtrls.cc hold the semaphore. */
      void setPartition();
    public:
      /** Return the Module. */
      Module& module();
      /** Return the semaphore taken around Module register access. */
      Semaphore& sem();
    public:
      /** Store the L0 select mode (FixedRate, ACRate or Sequence) used by setL0Select(). */
      void l0Select  (unsigned v);
      /** Store the rate that setL0Select() passes to Module::setL0Select_FixedRate(). */
      void fixedRate (unsigned v);
      /** Store the rate that setL0Select() passes to Module::setL0Select_ACRate(). */
      void acRate    (unsigned v);
      /** Store the timeslot mask that setL0Select() passes to Module::setL0Select_ACRate(). */
      void acTimeslot(unsigned v);
      /** Store the sequencer number that setL0Select() passes to Module::setL0Select_Sequence(). */
      void seqIdx    (unsigned v);
      /** Store the sequence bit that setL0Select() passes to Module::setL0Select_Sequence(). */
      void seqBit    (unsigned v);
      /** Store the destination mode that setL0Select() passes to Module::setL0Select_Destn(). */
      void dstSelect (unsigned v);
      /** Store the destination mask that setL0Select() passes to Module::setL0Select_Destn(). */
      void dstMask   (unsigned v);
      /** Store the key that msg_config() writes as the message payload. */
      void configKey (unsigned v);

      /** Under the semaphore, program the rate selection of this partition from the stored mode and values, then its destination selection. An unknown mode only prints a message; the destination is still written. */
      void setL0Select ();
      /** Declared but not defined in PVPCtrls.cc. */
      void setDstSelect();
      /** Under the semaphore, write the configuration key as the message payload, set the message header to TransitionId::Configure and insert the message. */
      void msg_config  ();
      /** Under the semaphore, set the message header to TransitionId::Enable and insert the message; the payload is not changed. */
      void msg_enable  ();
      /** Under the semaphore, set the message header to TransitionId::Disable and insert the message; the payload is not changed. */
      void msg_disable ();
      /** Under the semaphore, write payload 0, set the message header to 0 (the file-local MsgClear) and insert the message. */
      void msg_clear   ();
      /** Print the stored partition and L0 select values if PVPCtrls.cc is compiled with DBUG; otherwise does nothing. */
      void dump() const;
    public:
      /** L0 select modes stored by l0Select() and used by setL0Select(). */
      enum { FixedRate, /**< Use Module::setL0Select_FixedRate(). */ ACRate, /**< Use Module::setL0Select_ACRate(). */ Sequence  /**< Use Module::setL0Select_Sequence(). */ };
    private:
      std::vector<Pds_Epics::PVBase*> _pv;
      Module&    _m;
      Semaphore& _sem;
      PVPStats&  _stats;
      unsigned _shelf;
      unsigned _partition;
      bool     _enabled;
      unsigned _l0Select;
      unsigned _fixedRate;
      unsigned _acRate;
      unsigned _acTimeslot;
      unsigned _seqIdx;
      unsigned _seqBit;
      unsigned _dstSelect;
      unsigned _dstMask;
      unsigned _cfgKey;
    };
  };
};

#endif
