#include "psdaq/xpm/PVCtrls.hh"
#include "psdaq/xpm/PVSeq.hh"
#include "psdaq/xpm/Module.hh"
#include "psdaq/xpm/XpmSequenceEngine.hh"
#include "psdaq/app/AppUtils.hh"
#include "psdaq/service/Semaphore.hh"

#include <cpsw_error.h>  // To catch a CPSW exception and continue

#include <sstream>

#include <stdio.h>
#include <unistd.h>
#include <arpa/inet.h>

using Pds_Epics::EpicsPVA;
using Pds_Epics::PVMonitorCb;
using Pds::Xpm::MmcmPhaseLock;

static void fillMmcm( EpicsPVA*& pv, MmcmPhaseLock& mmcm ) {
  if (pv && pv->connected()) {
    unsigned w = 2048;
    pvd::shared_vector<int> vec(w+1);
    vec[0] = mmcm.delayValue;
    for(unsigned j=1; j<=w; j++) {
      mmcm.ramAddr = j;
      unsigned q = mmcm.ramData;
      vec[j] = q;
    }
    pv->putFromVector<int>(freeze(vec));
    pv = 0;
  }
}

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

    CPV(LinkRxTimeOut,  { GPVG(linkRxTimeOut(_idx, getScalarAs<unsigned>()))       },
                        { GPVP(linkRxTimeOut(_idx));                   })
    CPV(LinkGroupMask,  { GPVG(linkGroupMask(_idx, getScalarAs<unsigned>()))       },
                        { GPVP(linkGroupMask(_idx));                   })
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

    CPV(ResetMmcm0 ,    { if (getScalarAs<unsigned>()) _ctrl.resetMmcm(0); }, {})
    CPV(ResetMmcm1 ,    { if (getScalarAs<unsigned>()) _ctrl.resetMmcm(1); }, {})
    CPV(ResetMmcm2 ,    { if (getScalarAs<unsigned>()) _ctrl.resetMmcm(2); }, {})
    CPV(ResetMmcm3 ,    { if (getScalarAs<unsigned>()) _ctrl.resetMmcm(3); }, {})

    static PVCtrls* _instance = 0;

    PVCtrls::PVCtrls(Module& m, Semaphore& sem) : _pv(0), _m(m), _sem(sem), _seq(0), _seq_pv(0) 
    { _instance = this; }

    PVCtrls::~PVCtrls() {}

    void PVCtrls::allocate(const std::string& title)
    {
      for(unsigned i=0; i<_pv.size(); i++)
        delete _pv[i];
      _pv.resize(0);

      for(unsigned i=0; i<_seq_pv.size(); i++)
        delete _seq_pv[i];
      _seq_pv.resize(0);

      //  Wait for module to become ready
      { 
        _nmmcm = 0;
        if (Module::feature_rev()>0) {
          _nmmcm = 4;
          while(!_m._mmcm_amc.ready()) {
            printf("Waiting for XTPG phase lock: ready=[%c/%c]%c%c%c%c\n",
                   _m._usTiming.RxRstDone==1 ? 'T':'F',
                   _m._cuTiming.RxRstDone==1 ? 'T':'F',
                   _m._mmcm[0].ready() ? 'T':'F',
                   _m._mmcm[1].ready() ? 'T':'F',
                   _m._mmcm[2].ready() ? 'T':'F',
                   _m._mmcm_amc.ready()? 'T':'F');
            sleep(1);
          }

          for(unsigned i=0; i<4; i++) {
            std::stringstream ostr;
            ostr << title << ":XTPG:MMCM" << i;
            printf("mmcmpv[%d]: %s\n", i, ostr.str().c_str());
            _mmcmPV[i] = new EpicsPVA(ostr.str().c_str());  
          }

          for(unsigned i=0; i<_nmmcm; i++) 
            fillMmcm(_mmcmPV[i], i<3 ? _m._mmcm[i] : _m._mmcm_amc);
        }
      }

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

      NPVN( LinkRxTimeOut,      24    );
      NPVN( LinkGroupMask,      24    );
      NPVN( LinkTrgSrc,         24    );
      NPVN( LinkLoopback,       24    );
      NPVN( TxLinkReset,        24    );
      NPVN( RxLinkReset,        24    );
      NPVN( RxLinkDump,         Module::NDSLinks    );

      NPV( GroupL0Reset                             );
      NPV( GroupL0Enable                            );
      NPV( GroupL0Disable                           );
      NPV( GroupMsgInsert                           );

      for(unsigned i=0; i<1; i++) {
        std::ostringstream str; str << pvbase << "SEQENG:" << i;
        _seq_pv.push_back( new PVSeq(_m.sequenceEngine(), str.str()) );
      }

      o << "XTPG:";
      pvbase = o.str();
      NPV ( CuInput                                 );
      NPV ( CuDelay                                 );
      NPV ( CuBeamCode                              );
      NPV ( ClearErr                                );
      NPV ( ResetMmcm0                              );
      NPV ( ResetMmcm1                              );
      NPV ( ResetMmcm2                              );
      NPV ( ResetMmcm3                              );
      //  Always enable, use LinkGroupMask to exclude
      for(unsigned i=0; i<24; i++)
        _m.linkEnable(i,true);

      return;

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
    }

    void PVCtrls::checkPoint(unsigned iseq, unsigned addr)
    { _seq_pv[iseq]->checkPoint(addr); }

    void PVCtrls::resetMmcm(unsigned i)
    {
      if (Module::feature_rev()==0) 
        return;
      MmcmPhaseLock& mmcm = i<3 ? _m._mmcm[i] : _m._mmcm_amc;
      mmcm.reset();
      while(!_m._mmcm_amc.ready())
        sleep(1);
      for(unsigned i=0; i<_nmmcm; i++)
        fillMmcm(_mmcmPV[i], i<3 ? _m._mmcm[i] : _m._mmcm_amc);
    }

    Module& PVCtrls::module() { return _m; }
    Semaphore& PVCtrls::sem() { return _sem; }
    XpmSequenceEngine* PVCtrls::seq() { return _seq; }

    void* PVCtrls::notify_thread(void* arg)
    {
      const char* ip = (const char*)arg;
      unsigned uip = Psdaq::AppUtils::parse_ip(ip);
      unsigned port = 8197;

      int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
      if (fd < 0) {
        perror("Open socket");
        return 0;
      }

      sockaddr_in saddr;
      saddr.sin_family      = PF_INET;
      saddr.sin_addr.s_addr = htonl(uip);
      saddr.sin_port        = htons(port);

      if (connect(fd, (sockaddr*)&saddr, sizeof(saddr)) < 0) {
        perror("Error connecting UDP socket");
        return 0;
      }

      sockaddr_in haddr;
      socklen_t   haddr_len = sizeof(haddr);
      if (getsockname(fd, (sockaddr*)&haddr, &haddr_len) < 0) {
        perror("Error retrieving local address");
        return 0;
      }

      // "connect" to the sending socket
      char buf[4];
      send(fd,buf,sizeof(buf),0);

      uint8_t ibuf[256];
      ssize_t v;

      while ((v=read(fd, ibuf, sizeof(ibuf) ))>=0) {
        const uint16_t* p = reinterpret_cast<const uint16_t*>(ibuf);
        uint16_t mask = *p++;
        for(unsigned i=0; mask; i++, mask>>=1)
          if (mask & 1)
            _instance->checkPoint(i,*p++);
      }
      return 0;
    }
  };
};
