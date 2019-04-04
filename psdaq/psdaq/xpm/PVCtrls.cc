#include "psdaq/xpm/PVCtrls.hh"
#include "psdaq/xpm/Module.hh"
#include "psdaq/xpm/XpmSequenceEngine.hh"
#include "psdaq/service/Semaphore.hh"

#include <cpsw_error.h>  // To catch a CPSW exception and continue

#include <sstream>

#include <stdio.h>

using Pds_Epics::EpicsPVA;
using Pds_Epics::PVMonitorCb;

namespace Pds {
  namespace Xpm {

#define Q(a,b)      a ## b
#define PV(name)    Q(name, PV)

#define PVG(i) {                                                \
      try { _ctrl.module().i; }                                 \
      catch (CPSWError& e) { printf("cpsw exception %s\n",e.what()); } }
#define GPVG(i)   {                                             \
      _ctrl.sem().take();                                       \
      try { _ctrl.module().i; }                                 \
      catch (CPSWError& e) { printf("cpsw exception %s\n",e.what()); } \
      _ctrl.sem().give(); }
#define PVP(i) {                                                        \
    try { putFrom<unsigned>(_ctrl.module().i); }                        \
    catch (CPSWError& e) { printf("cpsw exception %s\n",e.what()); } }
#define GPVP(i) {                                               \
    _ctrl.sem().take();                                         \
    try { unsigned v = _ctrl.module().i;                        \
      putFrom<unsigned>(v); }                                   \
    catch (CPSWError& e) { printf("cpsw exception %s\n",e.what()); } \
    _ctrl.sem().give(); }

#define CPV(name, updatedBody, connectedBody)                           \
                                                                        \
    class PV(name) : public EpicsPVA,                                   \
                     public PVMonitorCb                                 \
    {                                                                   \
    public:                                                             \
      PV(name)(PVCtrls& ctrl, const char* pvName, unsigned idx = 0) :   \
        EpicsPVA(pvName, this),                                         \
        _ctrl(ctrl),                                                    \
        _idx(idx) {}                                                    \
      virtual ~PV(name)() {}                                            \
    public:                                                             \
      void updated  () { updatedBody }                                  \
      void onConnect() { connectedBody }                                \
    private:                                                            \
      PVCtrls& _ctrl;                                                   \
      unsigned _idx;                                                    \
    };

    CPV(LinkTxDelay,    { GPVG(linkTxDelay  (_idx, getScalarAs<unsigned>()))       },
                        { GPVP(linkTxDelay  (_idx));                   })
    CPV(LinkRxTimeOut,  { GPVG(linkRxTimeOut(_idx, getScalarAs<unsigned>()))       },
                        { GPVP(linkRxTimeOut(_idx));                   })
    CPV(LinkPartition,  { GPVG(linkPartition(_idx, getScalarAs<unsigned>()))       },
                        { GPVP(linkPartition(_idx));                   })
    CPV(LinkTrgSrc,     { GPVG(linkTrgSrc   (_idx, getScalarAs<unsigned>()))       },
                        { GPVP(linkTrgSrc   (_idx));                   })
    CPV(LinkLoopback,   { GPVG(linkLoopback (_idx, getScalarAs<unsigned>() != 0))  },
                        { GPVP(linkLoopback (_idx));                   })
    CPV(TxLinkReset,    { if (getScalarAs<unsigned>()!=0) GPVG(txLinkReset  (_idx)); },
                        {                                              })
    CPV(RxLinkReset,    { if (getScalarAs<unsigned>()!=0) GPVG(rxLinkReset  (_idx)); },
                        {                                              })
    CPV(RxLinkDump ,    { if (getScalarAs<unsigned>()!=0)
                            GPVG(rxLinkDump   (_idx));                 },
                        {                                              })
    CPV(LinkEnable,     { GPVG(linkEnable(_idx, getScalarAs<unsigned>() != 0));    },
                        { GPVP(linkEnable(_idx));                      })

    //    CPV(ModuleInit,     { GPVG(init     ());     }, { })
    CPV(DumpPll,        { GPVG(dumpPll  (_idx)); }, { })
    CPV(DumpTiming,     { PVG(dumpTiming(_idx)); }, { })
    CPV(DumpSeq,        { if (getScalarAs<unsigned>() && _ctrl.seq()) { _ctrl.seq()->dump();}}, {})
    CPV(SetVerbose,     { GPVG(setVerbose(getScalarAs<unsigned>())); }, {})
    CPV(TimeStampWr,    { if (getScalarAs<unsigned>()!=0) GPVG(setTimeStamp()); }, {})
    CPV(CuDelay    ,    { GPVG(setCuDelay   (getScalarAs<unsigned>())); }, {})
    CPV(CuInput    ,    { GPVG(setCuInput(getScalarAs<unsigned>())); }, {})
    CPV(CuBeamCode ,    { GPVG(setCuBeamCode(getScalarAs<unsigned>())); }, {})
    CPV(ClearErr   ,    { GPVG(clearCuFiducialErr(getScalarAs<unsigned>())); }, {})
    CPV(GroupL0Reset,     { GPVG(groupL0Reset(getScalarAs<unsigned>())); }, { })
    CPV(GroupL0Enable,    { GPVG(groupL0Enable(getScalarAs<unsigned>())); }, { })
    CPV(GroupL0Disable,   { GPVG(groupL0Disable(getScalarAs<unsigned>())); }, { })
    CPV(GroupMsgInsert,   { GPVG(groupMsgInsert(getScalarAs<unsigned>())); }, { })

    PVCtrls::PVCtrls(Module& m, Semaphore& sem) : _pv(0), _m(m), _sem(sem), _seq(0) {}
    PVCtrls::~PVCtrls() {}

    void PVCtrls::allocate(const std::string& title)
    {
      for(unsigned i=0; i<_pv.size(); i++)
        delete _pv[i];
      _pv.resize(0);

      std::ostringstream o;
      o << title << ":";
      std::string pvbase = o.str();

#define NPV(name)  _pv.push_back( new PV(name)(*this, (pvbase+#name).c_str()) )
#define NPVN(name, n)                                                        \
      for (unsigned i = 0; i < n; ++i) {                                     \
        std::ostringstream str;  str << i;  std::string idx = str.str();     \
        _pv.push_back( new PV(name)(*this, (pvbase+#name+idx).c_str(), i) ); \
      }

      //      NPV ( ModuleInit                              );
      NPVN( DumpPll,            Module::NAmcs       );
      NPVN( DumpTiming,         2                   );
      NPV ( DumpSeq                                 );
      NPV ( SetVerbose                              );
      NPV ( TimeStampWr                             );

      NPVN( LinkTxDelay,        24    );
      NPVN( LinkRxTimeOut,      24    );
      NPVN( LinkPartition,      24    );
      NPVN( LinkTrgSrc,         24    );
      NPVN( LinkLoopback,       24    );
      NPVN( TxLinkReset,        24    );
      NPVN( RxLinkReset,        24    );
      NPVN( RxLinkDump,         Module::NDSLinks    );
      NPVN( LinkEnable,         24    );

      NPV( GroupL0Reset                             );
      NPV( GroupL0Enable                            );
      NPV( GroupL0Disable                           );
      NPV( GroupMsgInsert                           );

      o << "XTPG:";
      pvbase = o.str();
      NPV ( CuInput                                 );
      NPV ( CuDelay                                 );
      NPV ( CuBeamCode                              );
      NPV ( ClearErr                                );

      //
      // Program sequencer
      //
      try {
        XpmSequenceEngine& engine = _m.sequenceEngine();
        engine.verbosity(2);
        // Setup a 22 pulse sequence to repeat 40000 times each second
        const unsigned NP = 22;
        std::vector<TPGen::Instruction*> seq;
        seq.push_back(new TPGen::FixedRateSync(6,1));  // sync start to 1Hz
        for(unsigned i=0; i<NP; i++) {
          unsigned bits=0;
          for(unsigned j=0; j<16; j++)
            if ((i*(j+1))%NP < (j+1))
              bits |= (1<<j);
          seq.push_back(new TPGen::ExptRequest(bits));
          seq.push_back(new TPGen::FixedRateSync(0,1)); // next pulse
        }
        seq.push_back(new TPGen::Branch(1, TPGen::ctrA,199));
        //  seq.push_back(new TPGen::Branch(1, TPGen::ctrB, 99));
        seq.push_back(new TPGen::Branch(1, TPGen::ctrB,199));
        seq.push_back(new TPGen::Branch(0));
        int rval = engine.insertSequence(seq);
        if (rval < 0)
          printf("Insert sequence failed [%d]\n", rval);
        engine.dump  ();
        engine.enable(true);
        engine.setAddress(rval,0,0);
        engine.reset ();
        _seq = &engine;
      } catch (CPSWError& e) { 
        _seq = 0;
        printf("cpsw exception %s\n",e.what()); 
        printf("Sequencer programming aborted\n");
      }
    }

    void PVCtrls::dump() const
    {
      _sem.take();
      unsigned enable=0;
      for(unsigned i=0; i<16; i++)
        if (_m.linkEnable(i))
          enable |= (1<<i);
      _sem.give();
    }

    Module& PVCtrls::module() { return _m; }
    Semaphore& PVCtrls::sem() { return _sem; }
    XpmSequenceEngine* PVCtrls::seq() { return _seq; }
  };
};
