#include "PV134Stats.hh"
#include "Module134.hh"
#include "FexCfg.hh"
#include "Pgp.hh"
#include "Jesd204b.hh"
#include "ChipAdcCore.hh"
#include "OptFmc.hh"

#include "psdaq/epicstools/EpicsPVA.hh"
using Pds_Epics::EpicsPVA;

#include <algorithm>
#include <sstream>
#include <cctype>
#include <string>
#include <vector>

#include <unistd.h>
#include <stdio.h>

#define LANES 8
#define CHANS 2
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
           _monJesd,
           _monEnv,
           _monAdc,
           _NumberOf };

    static const unsigned _sz_monTiming[] = {0};
    static const unsigned _sz_monPgp   [] = {4,4,4,4,4,4,4,4,4,4,};
    static const unsigned _sz_monRawBuf[] = {0};
    static const unsigned _sz_monFexBuf[] = {0};
    static const unsigned _sz_monRawDet[] = {16,16,16,16,};
    static const unsigned _sz_monJesd  [] = {112,4,};
    static const unsigned _sz_monEnv   [] = {0};
    static const unsigned _sz_monAdc   [] = {0};

    PV134Stats::PV134Stats(Module134& m) : _m(m), _pgp(m.pgp()), _pv(2*_NumberOf) {}
    PV134Stats::~PV134Stats() {}

    void PV134Stats::allocate(const std::string& title) {
      for(unsigned i=0; i<_pv.size(); i++)
        if (_pv[i]) {
          delete _pv[i];
          _pv[i]=0;
        }

      std::string pvbase_a = title + ":A:";
      std::string pvbase_b = title + ":B:";
      
#define PV_ADD(name  ) {                                                \
        _pv[_##name            ] = new EpicsPVA(STOU(pvbase_a + #name).c_str()); \
        _pv[_##name + _NumberOf] = new EpicsPVA(STOU(pvbase_b + #name).c_str()); }
      PV_ADD (monTiming);
      PV_ADD (monPgp);
      PV_ADD (monRawBuf);
      PV_ADD (monFexBuf);
      PV_ADD (monRawDet);
      PV_ADD (monJesd);
      PV_ADD (monEnv);
      PV_ADD (monAdc);
#undef PV_ADD

      printf("PVs allocated\n");

      _m.mon_start();
    }

    void PV134Stats::update()
    {
#define PVPUT(p) {                                              \
        Pds_Epics::EpicsPVA& pv = *_pv[_##p+i*_NumberOf];       \
        _v_##p[i] = v;                                          \
          if (pv.connected())                                   \
            pv.putFromStructure(&_v_##p[i],_sz_##p);            \
      }

      for(unsigned i=0; i<2; i++) {
        ChipAdcCore& chip = _m.chip(i);
        ChipAdcReg&  reg  = chip.reg;
        FexCfg&      fex  = chip.fex;

        { MonTiming v;
          v.timframecnt = DELT(reg.countEnable , _p_monTiming[i].timframeprv);
          v.timpausecnt = DELT(reg.countInhibit, _p_monTiming[i].timpauseprv);
          v.trigcntsum  = reg.countAcquire;
          v.readcntsum  = reg.countRead;
          v.startcntsum = reg.countStart;
          v.queuecntsum = reg.countQueue;
          v.trigcnt     = v.trigcntsum - _v_monTiming[i].trigcntsum;
          v.msgdelayset = reg.msgDelay&0xff;
          v.msgdelayget = (reg.msgDelay>>16)&0xff;
          v.headercntl0 = reg.headerCnt&0xffff;
          v.headercntof = (reg.headerCnt>>24)&0xff;
          v.headerfifow = (reg.headerFifo>> 0)&0xf;
          v.headerfifor = (reg.headerFifo>> 4)&0xf;
          PVPUT(monTiming); }

        { MonPgp v;
          for(unsigned j=0; j<4; j++) {
            Pgp& pgp = *_pgp[j+4*i];
            v.loclinkrdy [j] = pgp.localLinkReady ()?1:0;
            v.remlinkrdy [j] = pgp.remoteLinkReady()?1:0;
            v.txclkfreq  [j] = pgp.txClkFreqMHz();
            v.rxclkfreq  [j] = pgp.rxClkFreqMHz();
            v.txcntsum   [j] = pgp.txCount();
            v.txcnt      [j] = v.txcntsum[j] - _v_monPgp[i].txcntsum[j];
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
        if ((reg.csr&0x10)==0) {
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

        // JESD Status
        { MonJesd v;
          for(unsigned j=0; j<8; j++) 
            reinterpret_cast<Jesd204bStatus*>(v.stat)[j] = _m.jesd(i).status(j);
          for(unsigned j=0; j<4; j++)
            v.clks[j] = float(_m.optfmc().clks[j+1]&0x1fffffff)*1.e-6;
          PVPUT(monJesd); 
        }

        // ADC Monitoring
        { MonAdc v;
          v.oor_ina_0 = _m.optfmc().adcOutOfRange[i*4+0];
          v.oor_ina_1 = _m.optfmc().adcOutOfRange[i*4+1];
          v.oor_inb_0 = _m.optfmc().adcOutOfRange[i*4+2];
          v.oor_inb_1 = _m.optfmc().adcOutOfRange[i*4+3];
          v.alarm     = _m.optfmc().adcOutOfRange[i*1+8];
          PVPUT(monAdc); }
      }

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
        unsigned i;
        i = 0; PVPUT(monEnv);
        i = 1; PVPUT(monEnv); }

      /*
      HdrFifo* hdrf = _m.hdrFifo();
      PVPUTAU ( WrFifoCnt , FIFOS, (hdrf[i]._wrFifoCnt&0xf) );
      PVPUTAU ( RdFifoCnt , FIFOS, (hdrf[i]._rdFifoCnt&0xf) );
      */
    
#undef PVPUT
    }
  };
};
