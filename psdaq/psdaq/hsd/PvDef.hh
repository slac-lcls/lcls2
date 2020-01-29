#ifndef PvDef_hh
#define PvDef_hh

#include <cstdint>

namespace Pds {
  namespace HSD {
    // PV structures - must match the server's definition (see pvdef.py)
    class MonTiming {
    public:
      int32_t timframecnt;
      int32_t timpausecnt;
      int32_t timerrcntsum;
      int32_t timrstcntsum;
      int32_t trigcnt;
      int32_t trigcntsum;
      int32_t readcntsum;
      int32_t startcntsum;
      int32_t queuecntsum;
      int32_t msgdelayset;
      int32_t msgdelayget;
      int32_t headercntl0;
      int32_t headercntof;
      int32_t headerfifow;
      int32_t headerfifor;
      int32_t fulltotrig;
      int32_t nfulltotrig;
    };

    class MonTimingCalc {
    public:
      uint32_t timframeprv;
      uint32_t timpauseprv;
      uint32_t timerrcntsum;
      uint32_t timrstcntsum;
    };

    class MonPgp {
    public:
      int32_t loclinkrdy [4];
      int32_t remlinkrdy [4];
      float   txclkfreq  [4];
      float   rxclkfreq  [4];
      int32_t txcnt      [4];
      int32_t txcntsum   [4];
      int32_t txerrcntsum[4];
      int32_t rxcnt      [4];
      int32_t rxlast     [4];
      int32_t rempause   [4];
      int32_t remlinkid  [4];
    };

    class MonBuf {
    public:
      int32_t freesz;
      int32_t freeevt;
      int32_t fifoof;
    };

    class MonBufDetail {
    public:
      int32_t bufstate[16];
      int32_t trgstate[16];
      int32_t bufbeg  [16];
      int32_t bufend  [16];
    };

    class MonFlow {
    public:
      uint32_t fmask;
      uint32_t fcurr;
      uint32_t frdy;
      uint32_t srdy;
      uint32_t mrdy;
      uint32_t raddr;
      uint32_t npend;
      uint32_t ntrig;
      uint32_t nread;
      uint32_t pkoflow;
      uint32_t oflow;
    };

    class MonEnv {
    public:
      float local12v;
      float edge12v;
      float aux12v;
      float fmc12v;
      float boardtemp;
      float local3v3;
      float local2v5;
      float totalpwr;
      float fmcpwr;
      int32_t sync_even;
      int32_t sync_odd;
    };

    class MonAdc {
    public:
      int32_t oor_ina_0;
      int32_t oor_ina_1;
      int32_t oor_inb_0;
      int32_t oor_inb_1;
      int32_t alarm;
    };

    class MonJesd {
    public:
      int32_t stat[112];
      float   clks[5];
    };

    static const unsigned _sz_monTiming[] = {0};
    static const unsigned _sz_monPgp   [] = {4,4,4,4,4,4,4,4,4,4,4,};
    static const unsigned _sz_monRawBuf[] = {0};
    static const unsigned _sz_monFexBuf[] = {0};
    static const unsigned _sz_monRawDet[] = {16,16,16,16,};
    static const unsigned _sz_monFlow  [] = {0};
    static const unsigned _sz_monJesd  [] = {112,5,};
    static const unsigned _sz_monEnv   [] = {0};
    static const unsigned _sz_monAdc   [] = {0};
  };
}

#endif
