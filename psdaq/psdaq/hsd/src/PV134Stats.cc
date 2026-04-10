#include "PV134Stats.hh"
#include "Module134.hh"
#include "FexCfg.hh"
#include "Pgp.hh"
#include "Jesd204b.hh"
#include "ChipAdcCore.hh"
#include "OptFmc.hh"
#include "Fmc134Ctrl.hh"
#include "I2c134.hh"

#include "psdaq/mmhw/TprCore.hh"
#include "psdaq/mmhw/TriggerEventManager2.hh"

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
           _monFexDet,
           _monFlow,
           _monJesd,
           _monEnv,
           _monAdc,
           _fexOor,
           _monLinkId,  // PADDR_U
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
      PV_ADD (monFexDet);
      PV_ADD (monFlow);
      PV_ADD (monJesd);
      PV_ADD (monEnv);
      PV_ADD (monAdc);
      PV_ADD (fexOor);
      _pv[_monLinkId            ] = new EpicsPVA(STOU(pvbase_a + "PADDR_U").c_str());
      _pv[_monLinkId + _NumberOf] = new EpicsPVA(STOU(pvbase_b + "PADDR_U").c_str());
#undef PV_ADD

      printf("PVs allocated\n");

      _m.mon_start();

      for(unsigned i=0; i<2; i++) {
        Mmhw::TprCore& tpr  = _m.tpr();
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

      bool jesdreset = false;

      //  Update the linkID
      unsigned upaddr = _m.remote_id();

      for(unsigned i=0; i<2; i++) {
        ChipAdcCore& chip  = _m.chip(i);
        ChipAdcReg&  reg   = chip.reg;
        FexCfg&      fex   = chip.fex;
        Mmhw::TprCore& tpr = _m.tpr();
        Pds::Mmhw::TriggerEventBuffer& teb = _m.tem().det(i);

        { Pds_Epics::EpicsPVA& pv = *_pv[_monLinkId+i*_NumberOf];
          if (pv.connected())
              pv.putFrom(upaddr);
        }

        { MonTiming v;
          v.timframecnt = DELT(reg.countEnable , _p_monTiming[i].timframeprv);
          v.timpausecnt = DELT(reg.countInhibit, _p_monTiming[i].timpauseprv);
          v.timerrcntsum= tpr.RxDecErrs + tpr.RxDspErrs - _p_monTiming[i].timerrcntsum;
          v.timrstcntsum= tpr.RxRstDone                 - _p_monTiming[i].timrstcntsum;
          v.trigcntsum  = reg.countAcquire;
          // v.readcntsum  = reg.countRead;
          // v.startcntsum = reg.countStart;
          // v.queuecntsum = reg.countQueue;
          v.trigcntrate = v.trigcntsum - _v_monTiming[i].trigcntsum;
          unsigned group = teb.group&0xf;
          v.group       = group;
          v.l0delay     = _m.tem().xma().messageDelay[group];
          v.hdrcount    = teb.l0Count;
          v.chndatapaus = (teb.status>>1)&0x1;
          v.hdrfifopaus = (teb.status>>3)&0x1;
          v.hdrfifoof   = (teb.status>>2)&0x1;
          v.hdrfifoofl  = (teb.status>>0)&0x1;
          v.hdrfifow    = (teb.status>>4)&0x1f;
          v.hdrfifor    = (teb.pauseThresh>>0)&0x1f;

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
            v.rxcntsum   [j] = pgp.rxCount();
            v.rxcnt      [j] = v.rxcntsum[j] - _v_monPgp[i].rxcntsum[j];
            v.rxerrcntsum[j] = pgp.rxErrCount();
            //            v.rxcnt      [j] = pgp.rxOpCodeCount();
            v.rxlast     [j] = pgp.rxOpCodeLast() & 0xff;
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

        { MonBufDetail v(reg,0);
          PVPUT(monRawDet); }

        { MonBufDetail v(reg,1);
          PVPUT(monFexDet); }

        { MonFlow v;
          v.pkoflow = fex._oflow&0xff;            // cntOflow
          uint32_t flowstatus = fex._flowstatus;
          v.fmask = (flowstatus>>0)&0x7;          // r.fexb
          v.fcurr = (flowstatus>>4)&0xf;          // r.fexn<<1 !
          v.frdy  = (flowstatus>>8)&0xf;          // tValid(i)
          v.srdy  = (flowstatus>>12)&0x1;         // ?
          v.mrdy  = (flowstatus>>13)&0x7;         // tRdy/tVal/tRdy
          v.raddr = (flowstatus>>16)&0xffff;      // rdaddr(r.fexn)
          uint32_t flowidxs = fex._flowidxs;
          v.npend = (flowidxs>> 0)&0x1f;          // r.npend
          v.ntrig = (flowidxs>> 5)&0x1f;          // r.ntrig
          v.nread = (flowidxs>>10)&0x1f;          // r.nread
          v.oflow = (flowidxs>>16)&0xffff;        // wraddr(r.fexn)
          uint32_t buildflow = reg.buildStatus;
          v.bstat = (buildflow>>0)&0xf;           // build state
          v.dumps = (buildflow>>4)&0xf;           // build dumps ! (dmaRst)
          v.bhdrv = (buildflow>>8)&1;             // event header valid
          v.bval  = (buildflow>>9)&1;             // r.master.tValid
          v.brdy  = (buildflow>>10)&1;            // dmaslave tReady
          PVPUT(monFlow); }

        // JESD Status
        { MonJesd v;
          Jesd204bStatus* stat = reinterpret_cast<Jesd204bStatus*>(v.stat);
          for(unsigned j=0; j<8; j++) {
            stat[j] = _m.jesd(i).status(j);
            if (stat[j].recvDataValid==0)
            // if (stat[j].syncDone==0)
                jesdreset = true;
          }
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
          //  FEX baseline correction - out of range
          v.oor_fex   = fex._stream[1].parms[15];
          PVPUT(monAdc);

          Pds_Epics::EpicsPVA& pv = *_pv[_fexOor+i*_NumberOf];
          if (pv.connected())
              pv.putFrom(v.oor_fex);
        }

      }  // for(unsigned i=0; i<2; i++)

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

      if (jesdreset) {
          printf("--jesdadcinit (auto)--\n");
          _m.i2c_lock(I2cSwitch::PrimaryFmc);
          _m.jesdctl().default_init(_m.i2c().fmc_cpld,0);
          _m.i2c_unlock();
      }
    
#undef PVPUT
    }
  };
};
