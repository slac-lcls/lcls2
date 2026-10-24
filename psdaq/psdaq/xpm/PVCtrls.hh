/**
 * @file
 * @brief PVCtrls, the module-wide EPICS control PVs of an XPM (links, group L0 control, XTPG settings, MMCM scans and the sequence engine PVs).
 */
#ifndef Xpm_PVCtrls_hh
#define Xpm_PVCtrls_hh

#include "psdaq/epicstools/EpicsPVA.hh"

#include <string>
#include <vector>

namespace Pds {

  class Semaphore;

  namespace Xpm {
    
    class Module;
    class XpmSequenceEngine;
    class PVSeq;

    /** Module-wide EPICS control PVs of an XPM, created by allocate(); most PV callbacks call the Module under the semaphore. The last constructed object receives the sequence checkpoint notifications of notify_thread(). */
    class PVCtrls
    {
    public:
      /** Store the module and semaphore and make this object the target of notify_thread(). */
      PVCtrls(Module&, Semaphore& sem);
      /** Does nothing; the PVs are not deleted. */
      ~PVCtrls();
    public:
      /** Thread body: connect a UDP socket to port 8197 of the XPM at the IP address string arg, send a 4-byte datagram, then read datagrams until read() fails. In each datagram, the first 16-bit word is a mask and, for each set bit i, the next 16-bit word is passed to checkPoint(i, word). Returns null. */
      static void* notify_thread(void*);
    public:
      /** Delete earlier PVs and create the control PVs named title, a colon and the control name, plus one PVSeq under SEQENG:0 for a new Module::sequenceEngine(). If Module::feature_rev() is above 0, first wait until the AMC MMCM is ready and publish the delay scan of each MMCM whose XTPG:MMCMn PV is already connected. Then enable all 24 links; the sequencer programming code after the return statement is never reached. */
      void allocate(const std::string& title);
      /** Declared but not defined in PVCtrls.cc. */
      void update();
      /** Does nothing; empty body. */
      void dump() const;
      /** Forward addr to PVSeq::checkPoint() of sequence engine PV set iseq (no range check). */
      void checkPoint(unsigned iseq, unsigned addr);
      /** Reset MMCM i (3 or above selects the AMC MMCM), wait until the AMC MMCM is ready, and publish the delay scans of MMCM PVs not yet written. Does nothing if Module::feature_rev() is 0. */
      void resetMmcm(unsigned);
    public:
      /** Return the Module. */
      Module& module();
      /** Return the semaphore taken around Module register access. */
      Semaphore& sem();
      /** Return the sequence engine pointer. It stays null, because the code in allocate() that sets it is after an unconditional return. */
      XpmSequenceEngine* seq();
    private:
      std::vector<Pds_Epics::EpicsPVA*> _pv;
      Module&              _m;
      Semaphore&           _sem;
      XpmSequenceEngine*   _seq;
      std::vector<PVSeq*>  _seq_pv;
      Pds_Epics::EpicsPVA* _mmcmPV[4];
      unsigned             _nmmcm;
    };
  };
};

#endif
