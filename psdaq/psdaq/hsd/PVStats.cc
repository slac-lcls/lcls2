#include "psdaq/hsd/PVStats.hh"
#include "psdaq/hsd/Module.hh"
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

using Pds_Epics::EpicsPVA;

namespace Pds {
  namespace HSD {

    //  enumeration of PV insert order below
    enum { _TimFrameCnt, _TimPauseCnt,
           _TrigCnt, _TrigCntSum, _ReadCntSum, _StartCntSum, _QueueCntSum,
           _MsgDelaySet, _MsgDelayGet, _HeaderCntL0, _HeaderCntOF,
           _PgpLocLinkRdy, _PgpRemLinkRdy,
           _PgpTxClkFreq, _PgpRxClkFreq,
           _PgpTxCnt, _PgpTxCntSum, _PgpTxErrCnt, _PgpRxCnt, _PgpRxLast, _PgpRemPause,
           _Raw_FreeBufSz, _Raw_FreeBufEvt,
           _Fex_FreeBufSz, _Fex_FreeBufEvt,
           _Nat_FreeBufSz, _Nat_FreeBufEvt,
           _Raw_BufState, _Raw_TrgState, _Raw_BufBeg, _Raw_BufEnd,
           _Local12V, _Edge12V, _Aux12V,
           _Fmc12V, _BoardTemp,
           _Local3_3V, _Local2_5V, _Local1_8V,
           _TotalPower, _FmcPower,
           _WrFifoCnt, _RdFifoCnt,
           _SyncE, _SyncO,
           _NumberOf };

    PVStats::PVStats(Module& m) : _m(m), _pgp(m.pgp()), _pv(_NumberOf), _v(_NumberOf*16) {}
    PVStats::~PVStats() {}

    void PVStats::allocate(const std::string& title) {
      for(unsigned i=0; i<_NumberOf; i++)
        if (_pv[i]) {
          delete _pv[i];
          _pv[i]=0;
        }

      std::ostringstream o;
      o << title << ":";
      std::string pvbase = o.str();

#define PV_ADD(name  ) { _pv[_##name] = new EpicsPVA(STOU(pvbase + #name).c_str()); }
#define PV_ADDV(name,n) { _pv[_##name] = new EpicsPVA(STOU(pvbase + #name).c_str(), n); }

      PV_ADD (TimFrameCnt);
      PV_ADD (TimPauseCnt);
      PV_ADD (TrigCnt);
      PV_ADD (TrigCntSum);
      PV_ADD (ReadCntSum);
      PV_ADD (StartCntSum);
      PV_ADD (QueueCntSum);
      PV_ADD (MsgDelaySet);
      PV_ADD (MsgDelayGet);
      PV_ADD (HeaderCntL0);
      PV_ADD (HeaderCntOF);
      PV_ADDV(PgpLocLinkRdy,LANES);
      PV_ADDV(PgpRemLinkRdy,LANES);
      PV_ADDV(PgpTxClkFreq ,LANES);
      PV_ADDV(PgpRxClkFreq ,LANES);
      PV_ADDV(PgpTxCnt     ,LANES);
      PV_ADDV(PgpTxCntSum  ,LANES);
      PV_ADDV(PgpTxErrCnt  ,LANES);
      PV_ADDV(PgpRxCnt     ,LANES);
      PV_ADDV(PgpRxLast    ,LANES);
      PV_ADDV(Raw_FreeBufSz  ,16);
      PV_ADDV(Raw_FreeBufEvt ,16);
      PV_ADDV(Fex_FreeBufSz  ,16);
      PV_ADDV(Fex_FreeBufEvt ,16);
      //  Channel 0 stats only (by firmware)
      PV_ADDV(Raw_BufState   ,16);
      PV_ADDV(Raw_TrgState   ,16);
      PV_ADDV(Raw_BufBeg     ,16);
      PV_ADDV(Raw_BufEnd     ,16);

      PV_ADD(Local12V);
      PV_ADD(Edge12V);
      PV_ADD(Aux12V);
      PV_ADD(Fmc12V);
      PV_ADD(BoardTemp);
      PV_ADD(Local3_3V);
      PV_ADD(Local2_5V);
      PV_ADD(Local1_8V);
      PV_ADD(TotalPower);
      PV_ADD(FmcPower);

      PV_ADDV(WrFifoCnt      ,FIFOS);
      PV_ADDV(RdFifoCnt      ,FIFOS);

      PV_ADD(SyncE);
      PV_ADD(SyncO);
#undef PV_ADD
#undef PV_ADDV

      printf("PVs allocated\n");

      _m.mon_start();
    }

    void PVStats::update()
    {
#define PVPUTD(i,v)    {                                                \
        Pds_Epics::EpicsPVA& pv = *_pv[_##i];                           \
        if (pv.connected()) {                                           \
          pv.putFrom<double>(double(v));                                \
      } }
#define PVPUTU(i,v)    {                                                \
        Pds_Epics::EpicsPVA& pv = *_pv[_##i];                           \
        if (pv.connected()) {                                           \
          pv.putFrom<unsigned>(unsigned(v));                            \
        } }
#define PVPUTAU(p,m,v) {                                                \
        Pds_Epics::EpicsPVA& pv = *_pv[_##p];                           \
        if (pv.connected()) {                                           \
          pvd::shared_vector<unsigned> vec;                       \
          for(unsigned i=0; i<m; i++)                                   \
            vec[i] = unsigned(v);                                       \
          pv.putFromVector<unsigned>(freeze(vec)); } }
#define PVPUTDU(i,v)    {                                               \
        Pds_Epics::EpicsPVA& pv = *_pv[_##i];                           \
        if (pv.connected()) {                                           \
          pv.putFrom<unsigned>(unsigned(v - _v[_##i].value[0]));        \
          _v[_##i].value[0] = v; } }
#define PVPUTDAU(p,m,v) {                                               \
        Pds_Epics::EpicsPVA& pv = *_pv[_##p];                           \
        if (pv.connected()) {                                           \
          pvd::shared_vector<unsigned> vec;                       \
          for(unsigned i=0; i<m; i++) {                                 \
            vec[i] = unsigned(v-_v[_##p].value[i]);                     \
            _v[_##p].value[i] = v; }                                    \
          pv.putFromVector<unsigned>(freeze(vec)); } }

      QABase& base = *reinterpret_cast<QABase*>((char*)_m.reg()+0x00080000);
      PVPUTDU ( TimFrameCnt, base.countEnable);
      PVPUTDU ( TimPauseCnt, base.countInhibit);
      PVPUTDU ( TrigCnt    , base.countAcquire);
      PVPUTU  ( TrigCntSum , base.countAcquire);
      PVPUTU  ( ReadCntSum , base.countRead);
      PVPUTU  ( StartCntSum, base.countStart);
      PVPUTU  ( QueueCntSum, base.countQueue);
      PVPUTU  ( MsgDelaySet, (base.msgDelay&0xff) );
      PVPUTU  ( MsgDelayGet, ((base.msgDelay>>16)&0xff) );
      PVPUTU  ( HeaderCntL0, (base.headerCnt&0xfffff) );
      PVPUTU  ( HeaderCntOF, ((base.headerCnt>>24)&0xff) );

      PVPUTAU  ( PgpLocLinkRdy, LANES, _pgp[i]->localLinkReady ()?1:0);
      PVPUTAU  ( PgpRemLinkRdy, LANES, _pgp[i]->remoteLinkReady()?1:0);
      PVPUTAU  ( PgpTxClkFreq , LANES, _pgp[i]->txClkFreqMHz());
      PVPUTAU  ( PgpRxClkFreq , LANES, _pgp[i]->rxClkFreqMHz());
      PVPUTAU  ( PgpTxCntSum  , LANES, _pgp[i]->txCount     () );
      PVPUTDAU ( PgpTxCnt     , LANES, _pgp[i]->txCount     () );
      PVPUTDAU ( PgpTxErrCnt  , LANES, _pgp[i]->txErrCount  () );
      PVPUTDAU ( PgpRxCnt     , LANES, _pgp[i]->rxOpCodeCount() );
      PVPUTAU  ( PgpRxLast    , LANES, _pgp[i]->rxOpCodeLast () );
      PVPUTAU  ( PgpRxLast    , LANES, _pgp[i]->remPause     () );

      if ((base.csr&0x10)==0) {
        FexCfg* fex = _m.fex();
        PVPUTAU ( Raw_FreeBufSz  , CHANS, ((fex[i]._base[0]._free>> 0)&0xffff) );
        PVPUTAU ( Raw_FreeBufEvt , CHANS, ((fex[i]._base[0]._free>>16)&0x1f) );
        PVPUTAU ( Fex_FreeBufSz  , CHANS, ((fex[i]._base[1]._free>> 0)&0xffff) );
        PVPUTAU ( Fex_FreeBufEvt , CHANS, ((fex[i]._base[1]._free>>16)&0x1f) );
        PVPUTAU ( Nat_FreeBufSz  , CHANS, ((fex[i]._base[2]._free>> 0)&0xffff) );
        PVPUTAU ( Nat_FreeBufEvt , CHANS, ((fex[i]._base[2]._free>>16)&0x1f) );
      }

      unsigned state[16], addr[16];
      for(unsigned i=0; i<16; i++) {
        base.cacheSel = i;
        usleep(1);
        state[i] = base.cacheState;
        addr [i] = base.cacheAddr;
      }

      PVPUTAU ( Raw_BufState   , 16, ((state[i]>>0)&0xf) );
      PVPUTAU ( Raw_TrgState   , 16, ((state[i]>>4)&0xf) );
      PVPUTAU ( Raw_BufBeg     , 16, ((addr[i]>> 0)&0xffff) );
      PVPUTAU ( Raw_BufEnd     , 16, ((addr[i]>>16)&0xffff) );

      Pds::HSD::EnvMon mon = _m.mon();
      PVPUTD  ( Local12V  , mon.local12v   );
      PVPUTD  ( Edge12V   , mon.edge12v    );
      PVPUTD  ( Aux12V    , mon.aux12v     );
      PVPUTD  ( Fmc12V    , mon.fmc12v     );
      PVPUTD  ( BoardTemp , mon.boardTemp  );
      PVPUTD  ( Local3_3V , mon.local3_3v  );
      PVPUTD  ( Local2_5V , mon.local2_5v  );
      PVPUTD  ( Local1_8V , mon.local1_8v  );
      PVPUTD  ( TotalPower, mon.totalPower );
      PVPUTD  ( FmcPower  , mon.fmcPower   );

      HdrFifo* hdrf = _m.hdrFifo();
      PVPUTAU ( WrFifoCnt , 4, (hdrf[i]._wrFifoCnt&0xf) );
      PVPUTAU ( RdFifoCnt , 4, (hdrf[i]._rdFifoCnt&0xf) );

      PVPUTU  ( SyncE     , _m.trgPhase()[0]);
      PVPUTU  ( SyncO     , _m.trgPhase()[1]);
#undef PVPUTDU
#undef PVPUTDAU
#undef PVPUTU
#undef PVPUTAU
    }
  };
};
