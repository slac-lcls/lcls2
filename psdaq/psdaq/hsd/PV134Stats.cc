#include "PV134Stats.hh"
#include "Module134.hh"
#include "FexCfg.hh"
#include "Pgp.hh"
#include "Jesd204b.hh"
#include "ChipAdcCore.hh"
#include "OptFmc.hh"
#include "TprCore.hh"

#include "psdaq/mmhw/TriggerEventManager.hh"

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
           _monFlow,
           _monJesd,
           _monEnv,
           _monAdc,
           _NumberOf };

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
      PV_ADD (monFlow);
      PV_ADD (monJesd);
      PV_ADD (monEnv);
      PV_ADD (monAdc);
#undef PV_ADD

      printf("PVs allocated\n");

      _m.mon_start();

      for(unsigned i=0; i<2; i++) {
        TprCore& tpr  = _m.tpr();
        _p_monTiming[i].timerrcntsum = tpr.RxDecErrs + tpr.RxDspErrs;
        _p_monTiming[i].timrstcntsum = tpr.RxRstDone;
      }
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
        TprCore& tpr  = _m.tpr();
        Pds::Mmhw::TriggerEventBuffer& teb = _m.tem().det(i);

        { MonTiming v;
          v.timframecnt = DELT(reg.countEnable , _p_monTiming[i].timframeprv);
          v.timpausecnt = DELT(reg.countInhibit, _p_monTiming[i].timpauseprv);
          v.timerrcntsum= tpr.RxDecErrs + tpr.RxDspErrs - _p_monTiming[i].timerrcntsum;
          v.timrstcntsum= tpr.RxRstDone                 - _p_monTiming[i].timrstcntsum;
          v.trigcntsum  = reg.countAcquire;
          v.readcntsum  = reg.countRead;
          // v.startcntsum = reg.countStart;
          // v.queuecntsum = reg.countQueue;
          v.trigcnt     = v.trigcntsum - _v_monTiming[i].trigcntsum;
          unsigned group = teb.group&0xf;
          v.msgdelayset = _m.tem().xma().messageDelay[group];
          //          v.msgdelayget = (reg.msgDelay>>16)&0xff;
          v.headercntl0 = teb.l0Count;
          v.headercntof = (teb.status>>0)&5;
          v.headerfifow = (teb.status>>3)&0x1f;
          //          v.headerfifor = (reg.headerFifo>> 4)&0xf;
          v.fulltotrig  = (teb.fullToTrig>> 0)&0xfff;
          v.nfulltotrig = (teb.nfullToTrig>>0)&0xfff;
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
            v.remlinkid  [j] = pgp.remoteLinkId();
          }
          PVPUT(monPgp);
        }

        //
        //  DmaClk must be running to read these registers
        //
        if ((reg.csr&0x10)==0) {
          { MonBuf v;
            fex._base[0].getFree(v.freesz,v.freeevt,v.fifoof);
            PVPUT(monRawBuf); }
          { MonBuf v;
            fex._base[1].getFree(v.freesz,v.freeevt,v.fifoof);
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
        
        // { MonFlow v;
        //   v.pkoflow = fex._oflow&0xff;
        //   uint32_t flowstatus = fex._flowstatus;
        //   v.fmask = (flowstatus>>0)&0x7;
        //   v.fcurr = (flowstatus>>4)&0xf;
        //   v.frdy  = (flowstatus>>8)&0xf;
        //   v.srdy  = (flowstatus>>12)&0x1;
        //   v.mrdy  = (flowstatus>>13)&0x7;
        //   v.raddr = (flowstatus>>16)&0xffff;
        //   uint32_t flowidxs = fex._flowidxs;
        //   v.npend = (flowidxs>> 0)&0x1f;
        //   v.ntrig = (flowidxs>> 5)&0x1f;
        //   v.nread = (flowidxs>>10)&0x1f;
        //   v.oflow = (flowidxs>>16)&0xff;
        //   PVPUT(monFlow); }

        // JESD Status
        { MonJesd v;
          for(unsigned j=0; j<8; j++) 
            reinterpret_cast<Jesd204bStatus*>(v.stat)[j] = _m.jesd(i).status(j);
          for(unsigned j=0; j<5; j++)
            v.clks[j] = float(_m.optfmc().clks[j]&0x1fffffff)*1.e-6;
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
