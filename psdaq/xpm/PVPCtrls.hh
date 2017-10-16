#ifndef Xpm_PVPCtrls_hh
#define Xpm_PVPCtrls_hh

#include <string>
#include <vector>

namespace Pds_Epics { class PVBase; }

namespace Pds {

  class Semaphore;

  namespace Xpm {

    class Module;

    class PVPCtrls
    {
    public:
      PVPCtrls(Module&,
               Semaphore&,
               unsigned partition);
      ~PVPCtrls();
    public:
      void allocate(const std::string& title);
      void enable(bool);
      void update();
      bool enabled() const;
      void setPartition();
    public:
      Module& module();
      Semaphore& sem();
    public:
      void l0Select  (unsigned v);
      void fixedRate (unsigned v);
      void acRate    (unsigned v);
      void acTimeslot(unsigned v);
      void seqIdx    (unsigned v);
      void seqBit    (unsigned v);
      void dstSelect (unsigned v);
      void dstMask   (unsigned v);
      void messageHdr(unsigned v);

      void setL0Select ();
      void setDstSelect();
      void messageIns  ();
      void dump() const;
    public:
      enum { FixedRate, ACRate, Sequence };
    private:
      std::vector<Pds_Epics::PVBase*> _pv;
      Module&  _m;
      Semaphore& _sem;
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
      unsigned _msgHdr;
    };
  };
};

#endif
