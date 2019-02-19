#include "psdaq/dti/PVCtrls.hh"
#include "psdaq/dti/Module.hh"

#include <cpsw_error.h>  // To catch a CPSW exception and continue

#include <sstream>

#include <stdio.h>

using Pds_Epics::EpicsPVA;
using Pds_Epics::PVMonitorCb;

namespace Pds {
  namespace Dti {

#define Q(a,b)      a ## b
#define PV(name)    Q(name, PV)

#define PRT(value)  printf("%60.60s: %32.32s: 0x%02x\n", __PRETTY_FUNCTION__, name().c_str(), value)

#define PVG(i)      PRT(getScalarAs<unsigned>()); _ctrl.module().i;
#define PVP(i)      PRT( _ctrl.module().i );  putFrom<unsigned>(_ctrl.module().i);

#define CPV(name, updatedBody, connectedBody)                           \
                                                                        \
    class PV(name) : public EpicsPVA,                                   \
                     public PVMonitorCb                                 \
    {                                                                   \
    public:                                                             \
      PV(name)(PVCtrls& ctrl, const char* pvName, unsigned idx = 0) :   \
        EpicsPVA(pvName, this),                                          \
        _ctrl(ctrl),                                                    \
        _idx(idx) {}                                                    \
      virtual ~PV(name)() {}                                            \
    public:                                                             \
      void updated();                                                   \
      void onConnect();                                             \
    public:                                                             \
      void put() { if (this->EpicsPVA::connected())  _channel.put(); }   \
    private:                                                            \
      PVCtrls& _ctrl;                                                   \
      unsigned _idx;                                                    \
    };                                                                  \
    void PV(name)::updated()                                            \
    {                                                                   \
      try { updatedBody }                                               \
      catch(CPSWError& e) {}                                            \
    }                                                                   \
    void PV(name)::onConnect()                                          \
    {                                                                   \
      connectedBody                                                     \
    }

    CPV(ModuleInit,     { PVG(init());        }, { })
    CPV(CountClear,     { PVG(clearCounters()); }, { })

    CPV(UsLinkTrigDelay,  { PVG(usLinkTrigDelay(_idx, getScalarAs<unsigned>()));    },
                          { PVP(usLinkTrigDelay(_idx));                 })

    CPV(UsLinkFwdMask,  { PVG(usLinkFwdMask(_idx, getScalarAs<unsigned>()));    },
                        { PVP(usLinkFwdMask(_idx));                 })

    CPV(UsLinkPartition,  { PVG(usLinkPartition(_idx, getScalarAs<unsigned>()));    },
                          { PVP(usLinkPartition(_idx));                 })

    PVCtrls::PVCtrls(Module& m) : _pv(0), _m(m) {}
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

      NPV ( ModuleInit                              );
      NPV ( CountClear                              );
      NPVN( UsLinkTrigDelay,    Module::NUsLinks    );
      NPVN( UsLinkFwdMask,      Module::NUsLinks    );
      NPVN( UsLinkPartition,    Module::NUsLinks    );

    }

    Module& PVCtrls::module() { return _m; }

  };
};
