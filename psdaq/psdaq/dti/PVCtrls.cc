#include "psdaq/dti/PVCtrls.hh"
#include "psdaq/dti/Module.hh"

#include <cpsw_error.h>  // To catch a CPSW exception and continue

#include <sstream>

#include <stdio.h>

using Pds_Epics::EpicsCA;
using Pds_Epics::PVMonitorCb;

namespace Pds {
  namespace Dti {

#define Q(a,b)      a ## b
#define PV(name)    Q(name, PV)

#define TOU(value)  *reinterpret_cast<unsigned*>(value)
#define TOB(value)  *reinterpret_cast<bool*>(value)
#define PRT(value)  printf("%60.60s: %32.32s: 0x%02x\n", __PRETTY_FUNCTION__, _channel.epicsName(), value)

#define PVG(i)      PRT(TOU(data()));    _ctrl.module().i;
#define PVP(i)      PRT( ( TOU(data()) = _ctrl.module().i ) );  put();

#define CPV(name, updatedBody, connectedBody)                           \
                                                                        \
    class PV(name) : public EpicsCA,                                    \
                     public PVMonitorCb                                 \
    {                                                                   \
    public:                                                             \
      PV(name)(PVCtrls& ctrl, const char* pvName, unsigned idx = 0) :   \
        EpicsCA(pvName, this),                                          \
        _ctrl(ctrl),                                                    \
        _idx(idx) {}                                                    \
      virtual ~PV(name)() {}                                            \
    public:                                                             \
      void updated();                                                   \
      void connected(bool);                                             \
    public:                                                             \
      void put() { if (this->EpicsCA::connected())  _channel.put(); }   \
    private:                                                            \
      PVCtrls& _ctrl;                                                   \
      unsigned _idx;                                                    \
    };                                                                  \
    void PV(name)::updated()                                            \
    {                                                                   \
      try { updatedBody }                                               \
      catch(CPSWError& e) {}                                            \
    }                                                                   \
    void PV(name)::connected(bool c)                                    \
    {                                                                   \
      this->EpicsCA::connected(c);                                      \
      connectedBody                                                     \
    }

    CPV(ModuleInit,     { PVG(init());        }, { })
    CPV(CountClear,     { PVG(clearCounters()); }, { })

    CPV(UsLinkTrigDelay,  { PVG(usLinkTrigDelay(_idx, TOU(data())));    },
                          { PVP(usLinkTrigDelay(_idx));                 })

    CPV(UsLinkFwdMask,  { PVG(usLinkFwdMask(_idx, TOU(data())));    },
                        { PVP(usLinkFwdMask(_idx));                 })

    CPV(UsLinkPartition,  { PVG(usLinkPartition(_idx, TOU(data())));    },
                          { PVP(usLinkPartition(_idx));                 })

    PVCtrls::PVCtrls(Module& m) : _pv(0), _m(m) {}
    PVCtrls::~PVCtrls() {}

    void PVCtrls::allocate(const std::string& title)
    {
      if (ca_current_context() == NULL) {
        printf("Initializing context\n");
        SEVCHK ( ca_context_create(ca_enable_preemptive_callback ),
                 "Calling ca_context_create" );
      }

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

      // Wait for monitors to be established
      ca_pend_io(0);
    }

    Module& PVCtrls::module() { return _m; }

  };
};
