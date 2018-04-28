#include "psdaq/hsd/PVStats.hh"
#include "psdaq/hsd/Module.hh"
#include "psdaq/hsd/FexCfg.hh"
#include "psdaq/hsd/HdrFifo.hh"
#include "psdaq/hsd/Pgp.hh"
#include "psdaq/hsd/QABase.hh"

#include "psdaq/epicstools/PVWriter.hh"
using Pds_Epics::PVWriter;

#include <algorithm>
#include <sstream>
#include <cctype>
#include <string>
#include <vector>

#include <unistd.h>
#include <stdio.h>

static std::string STOU(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c){ return std::toupper(c); }
                 );
  return s;
}

using Pds_Epics::PVWriter;

namespace Pds {
  namespace HSD {

    //  enumeration of PV insert order below
    enum { _TimFrameCnt, _TimPauseCnt,
           _PgpLocLinkRdy, _PgpRemLinkRdy,
           _PgpTxClkFreq, _PgpRxClkFreq,
           _PgpTxCnt, _PgpTxErrCnt, _PgpRxCnt, _PgpRxLast,
           _Raw_FreeBufSz, _Raw_FreeBufEvt,
           _Fex_FreeBufSz, _Fex_FreeBufEvt,
           _Raw_BufState, _Raw_TrgState, _Raw_BufBeg, _Raw_BufEnd,
           _TestPattErr, _TestPattBit,
           _BramWrErr, _BramWrSamp,
           _BramRdErr, _BramRdSamp,
           _WrFifoCnt, _RdFifoCnt,
           _NumberOf };

    PVStats::PVStats(Module& m) : _m(m), _pgp(m.pgp()), _pv(_NumberOf), _v(_NumberOf*16) {}
    PVStats::~PVStats() {}

    void PVStats::allocate(const std::string& title) {
      if (ca_current_context() == NULL) {
        printf("Initializing context\n");
        SEVCHK ( ca_context_create(ca_enable_preemptive_callback ),
                 "Calling ca_context_create" );
      }

      for(unsigned i=0; i<_NumberOf; i++)
        if (_pv[i]) {
          delete _pv[i];
          _pv[i]=0;
        }

      std::ostringstream o;
      o << title << ":";
      std::string pvbase = o.str();

#define PV_ADD(name  ) { _pv[_##name] = new PVWriter(STOU(pvbase + #name).c_str()); }
#define PV_ADDV(name,n) { _pv[_##name] = new PVWriter(STOU(pvbase + #name).c_str(), n); }

      PV_ADD (TimFrameCnt);
      PV_ADD (TimPauseCnt);
      PV_ADDV(PgpLocLinkRdy,4);
      PV_ADDV(PgpRemLinkRdy,4);
      PV_ADDV(PgpTxClkFreq ,4);
      PV_ADDV(PgpRxClkFreq ,4);
      PV_ADDV(PgpTxCnt     ,4);
      PV_ADDV(PgpTxErrCnt  ,4);
      PV_ADDV(PgpRxCnt     ,4);
      PV_ADDV(PgpRxLast    ,4);
      PV_ADDV(Raw_FreeBufSz  ,16);
      PV_ADDV(Raw_FreeBufEvt ,16);
      PV_ADDV(Fex_FreeBufSz  ,16);
      PV_ADDV(Fex_FreeBufEvt ,16);
      //  Channel 0 stats only (by firmware)
      PV_ADDV(Raw_BufState   ,16);
      PV_ADDV(Raw_TrgState   ,16);
      PV_ADDV(Raw_BufBeg     ,16);
      PV_ADDV(Raw_BufEnd     ,16);
      PV_ADDV(TestPattErr    ,4);
      PV_ADDV(TestPattBit    ,4);
      PV_ADDV(BramWrErr      ,4);
      PV_ADDV(BramWrSamp     ,4);
      PV_ADDV(BramRdErr      ,4);
      PV_ADDV(BramRdSamp     ,4);
      PV_ADDV(WrFifoCnt      ,4);
      PV_ADDV(RdFifoCnt      ,4);

#undef PV_ADD
#undef PV_ADDV

      ca_pend_io(0);
      printf("PVs allocated\n");
    }

    void PVStats::update()
    {
   
#define PVPUTU(i,v)    {                                                \
        Pds_Epics::PVWriter& pv = *_pv[_##i];                           \
        if (pv.connected()) {                                           \
          *reinterpret_cast<unsigned*>(pv.data()) = unsigned(v);        \
          pv.put(); } }
#define PVPUTAU(p,m,v) {                                                \
        Pds_Epics::PVWriter& pv = *_pv[_##p];                           \
        if (pv.connected()) {                                           \
          for(unsigned i=0; i<m; i++)                                   \
            reinterpret_cast<unsigned*>(pv.data())[i] = unsigned(v);    \
          pv.put(); } }
#define PVPUTDU(i,v)    {                                               \
        Pds_Epics::PVWriter& pv = *_pv[_##i];                           \
        if (pv.connected()) {                                           \
          *reinterpret_cast<unsigned*>(pv.data())    =                  \
            unsigned(v - _v[_##i].value[0]);                            \
          pv.put();                                                     \
          _v[_##i].value[0] = v; } }
#define PVPUTDAU(p,m,v) {                                               \
        Pds_Epics::PVWriter& pv = *_pv[_##p];                           \
        if (pv.connected()) {                                           \
          for(unsigned i=0; i<m; i++) {                                 \
            reinterpret_cast<unsigned*>(pv.data())[i] =                 \
              unsigned(v-_v[_##p].value[i]);                            \
            _v[_##p].value[i] = v; }                                    \
          pv.put(); } }
      
      QABase& base = *reinterpret_cast<QABase*>((char*)_m.reg()+0x00080000);
      PVPUTDU ( TimFrameCnt, base.countEnable);
      PVPUTDU ( TimPauseCnt, base.countInhibit);

      PVPUTAU  ( PgpLocLinkRdy, 4, _pgp[i]->localLinkReady ()?1:0);
      PVPUTAU  ( PgpRemLinkRdy, 4, _pgp[i]->remoteLinkReady()?1:0);
      PVPUTAU  ( PgpTxClkFreq , 4, _pgp[i]->txClkFreqMHz());
      PVPUTAU  ( PgpRxClkFreq , 4, _pgp[i]->rxClkFreqMHz());
      PVPUTDAU ( PgpTxCnt     , 4, _pgp[i]->txCount     () ); 
      PVPUTDAU ( PgpTxErrCnt  , 4, _pgp[i]->txErrCount  () );
      PVPUTDAU ( PgpRxCnt     , 4, _pgp[i]->rxOpCodeCount() );
      PVPUTAU  ( PgpRxLast    , 4, _pgp[i]->rxOpCodeLast () );

      FexCfg* fex = _m.fex();
      PVPUTAU ( Raw_FreeBufSz  , 4, ((fex[i]._base[0]._free>> 0)&0xffff) ); 
      PVPUTAU ( Raw_FreeBufEvt , 4, ((fex[i]._base[0]._free>>16)&0x1f) ); 
      PVPUTAU ( Fex_FreeBufSz  , 4, ((fex[i]._base[1]._free>> 0)&0xffff) ); 
      PVPUTAU ( Fex_FreeBufEvt , 4, ((fex[i]._base[1]._free>>16)&0x1f) ); 

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

      PVPUTAU ( TestPattErr , 4, (fex[i]._test_pattern_errors) );
      PVPUTAU ( TestPattBit , 4, (fex[i]._test_pattern_errbits) );

      PVPUTAU ( BramWrErr   , 4, (fex[i]._bram_wr_errors) );
      PVPUTAU ( BramWrSamp  , 4, (fex[i]._bram_wr_sample) );
      PVPUTAU ( BramRdErr   , 4, (fex[i]._bram_rd_errors) );
      PVPUTAU ( BramRdSamp  , 4, (fex[i]._bram_rd_sample) );

      HdrFifo* hdrf = _m.hdrFifo();
      PVPUTAU ( WrFifoCnt , 4, (hdrf[i]._wrFifoCnt&0xf) );
      PVPUTAU ( RdFifoCnt , 4, (hdrf[i]._rdFifoCnt&0xf) );
    
#undef PVPUTDU
#undef PVPUTDAU
#undef PVPUTU
#undef PVPUTAU

      ca_flush_io();
    }
  };
};
