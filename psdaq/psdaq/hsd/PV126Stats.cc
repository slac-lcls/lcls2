#include "psdaq/hsd/PV126Stats.hh"
#include "psdaq/hsd/Module126.hh"
#include "psdaq/hsd/FexCfg.hh"
#include "psdaq/hsd/HdrFifo.hh"
#include "psdaq/hsd/Pgp.hh"
#include "psdaq/hsd/QABase.hh"

#include "psdaq/epicstools/EpicsPVA.hh"
using Pds_Epics::EpicsPVA;

#include <algorithm>
#include <sstream>
#include <cctype>
#include <string>
#include <vector>

#include <unistd.h>
#include <stdio.h>

#define LANES 4
#define CHANS 1
#define FIFOS CHANS

static std::string STOU(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c){ return std::toupper(c); }
                 );
  return s;
}

static inline unsigned DELT(unsigned val, unsigned& prv)
{
  unsigned q = val-prv;
  prv = val;
  return q;
}

using Pds_Epics::EpicsPVA;

namespace Pds {
  namespace HSD {
    //  enumeration of PV insert order below
    enum { _monTiming,
           _monPgp,
           _monRawBuf,
           _monFexBuf,
           _monRawDet,
           _monEnv,
           _NumberOf };

    PV126Stats::PV126Stats(Module126& m) : _m(m), _pgp(m.pgp()), _pv(_NumberOf), _v(_NumberOf*16) {}
    PV126Stats::~PV126Stats() {}

    void PV126Stats::allocate(const std::string& title) {
      for(unsigned i=0; i<_NumberOf; i++)
        if (_pv[i]) {
          delete _pv[i];
          _pv[i]=0;
        }

      std::string pvbase = title + ":";

#define PV_ADD(name  ) { _pv[_##name] = new EpicsPVA(STOU(pvbase + #name).c_str()); }
      PV_ADD (monTiming);
      PV_ADD (monPgp);
      PV_ADD (monRawBuf);
      PV_ADD (monFexBuf);
      PV_ADD (monRawDet);
      PV_ADD (monEnv);
#undef PV_ADD

      printf("PVs allocated\n");

      _m.mon_start();
    }

    void PV126Stats::update()
    {
#define PVPUT(p) {                                      \
        Pds_Epics::EpicsPVA& pv = *_pv[_##p];           \
        _v_##p[0] = v;                                  \
          if (pv.connected())                           \
            pv.putFromStructure(&_v_##p[0],_sz_##p);    \
          else                                          \
            printf("%s not connected\n",#p);            \
      }
      
      QABase&  reg  = *reinterpret_cast<QABase*>((char*)_m.reg()+0x00080000);
      HdrFifo& hdrf = _m.hdrFifo()[0];
      FexCfg & fex  = _m.fex()[0];

      { MonTiming v;
        v.timframecnt = DELT(reg.countEnable , _p_monTiming[0].timframeprv);
        v.timpausecnt = DELT(reg.countInhibit, _p_monTiming[0].timpauseprv);
        v.trigcntsum  = reg.countAcquire;
        v.readcntsum  = reg.countRead;
        v.startcntsum = reg.countStart;
        v.queuecntsum = reg.countQueue;
        v.trigcnt     = v.trigcntsum - _v_monTiming[0].trigcntsum;
        v.msgdelayset = reg.msgDelay&0xff;
        v.msgdelayget = (reg.msgDelay>>16)&0xff;
        v.headercntl0 = reg.headerCnt&0xffff;
        v.headercntof = (reg.headerCnt>>24)&0xff;
        v.headerfifow = (hdrf._wrFifoCnt>> 0)&0xf;
        v.headerfifor = (hdrf._rdFifoCnt>> 0)&0xf;
        PVPUT(monTiming); }

      { MonPgp v;
        for(unsigned j=0; j<4; j++) {
          Pgp& pgp = *_pgp[j];
          v.loclinkrdy [j] = pgp.localLinkReady ()?1:0;
          v.remlinkrdy [j] = pgp.remoteLinkReady()?1:0;
          v.txclkfreq  [j] = pgp.txClkFreqMHz();
          v.rxclkfreq  [j] = pgp.rxClkFreqMHz();
          v.txcntsum   [j] = pgp.txCount();
          v.txcnt      [j] = v.txcntsum[j] - _v_monPgp[0].txcntsum[j];
          v.txerrcntsum[j] = pgp.txErrCount();
          v.rxcnt      [j] = pgp.rxOpCodeCount();
          v.rxlast     [j] = pgp.rxOpCodeLast();
          v.rempause   [j] = pgp.remPause();
        }
        PVPUT(monPgp);
      }

      //
      //  DmaClk must be running to read these registers
      //
      //      if ((reg.csr&0x10)==0) {
      if (1) {
        { MonBuf v;
          v.freesz  = (fex._base[0]._free>> 0)&0xffff;
          v.freeevt = (fex._base[0]._free>>16)&0x1f;
          v.fifoof  = (fex._base[0]._free>>24)&0xff;
          PVPUT(monRawBuf); }
        { MonBuf v;
          v.freesz  = (fex._base[1]._free>> 0)&0xffff;
          v.freeevt = (fex._base[1]._free>>16)&0x1f;
          v.fifoof  = (fex._base[1]._free>>24)&0xff;
          PVPUT(monFexBuf); }
      }

      { MonBufDetail v;
        for(unsigned j=0; j<16; j++) {
          reg.cacheSel = j;
          usleep(1);
          unsigned state = reg.cacheState;
          unsigned addr  = reg.cacheAddr;
          v.bufstate[j] = (state>>0)&0xf;
          v.trgstate[j] = (state>>4)&0xf;
          v.bufbeg  [j] = (addr >> 0)&0xffff;
          v.bufend  [j] = (addr >>16)&0xffff;
        }
        PVPUT(monRawDet); }

      { MonEnv v;
        EnvMon mon = _m.mon();
        v.local12v = mon.local12v;
        v.edge12v  = mon.edge12v;
        v.aux12v   = mon.aux12v;
        v.fmc12v   = mon.fmc12v;
        v.boardtemp= mon.boardTemp;
        v.local3v3 = mon.local3_3v;
        v.local2v5 = mon.local2_5v;
        v.totalpwr = mon.totalPower;
        v.fmcpwr   = mon.fmcPower;
        v.sync_even= _m.trgPhase()[0];
        v.sync_odd = _m.trgPhase()[1];
        PVPUT(monEnv); }

#undef PVPUT
    }
  };
};
