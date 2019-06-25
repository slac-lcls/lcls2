#ifndef Hsd_PV134Stats_hh
#define Hsd_PV134Stats_hh

#include <string>
#include <vector>

namespace Pds_Epics {
  class EpicsPVA;
};

namespace Pds {
  namespace HSD {
    class Pgp;
    class Module134;

    // PV structures - must match the server's definition
    class MonTiming {
    public:
      int32_t timframecnt;
      int32_t timpausecnt;
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
    };

    class MonTimingCalc {
    public:
      uint32_t timframeprv;
      uint32_t timpauseprv;
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
      float   clks[4];
    };

    class PV134Stats {
    public:
      PV134Stats(Module134&);
      ~PV134Stats();
    public:
      void allocate(const std::string& title);
      void update();
    private:
      Module134& _m;
      std::vector<Pgp*> _pgp;
      std::vector<Pds_Epics::EpicsPVA*> _pv;

      MonTiming     _v_monTiming[2];
      MonPgp        _v_monPgp   [2];
      MonBuf        _v_monRawBuf[2];
      MonBuf        _v_monFexBuf[2];
      MonBufDetail  _v_monRawDet[2];
      MonEnv        _v_monEnv   [2];
      MonAdc        _v_monAdc   [2];
      MonJesd       _v_monJesd  [2];

      MonTimingCalc _p_monTiming[2];

      class PVCache {
      public:
        unsigned value[4];
      };
      std::vector<PVCache> _v;
    };
  };
};

#endif
