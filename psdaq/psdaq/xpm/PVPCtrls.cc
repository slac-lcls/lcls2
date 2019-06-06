#include "psdaq/xpm/PVPCtrls.hh"
#include "psdaq/xpm/Module.hh"
#include "psdaq/xpm/PVPStats.hh"
#include "psdaq/epicstools/PVBase.hh"
#include "psdaq/service/Semaphore.hh"
#include "psdaq/cphw/Logging.hh"

#include <cpsw_error.h>  // To catch a CPSW exception and continue

#include <string>
#include <sstream>

#include <stdio.h>
#include "xtcdata/xtc/TransitionId.hh"

//#define DBUG

using Pds::Cphw::Logger;
using Pds_Epics::PVBase;
using namespace XtcData;

//  XPM Message Enum
enum { MsgClear =0,
       MsgDelay =1,
       MsgConfig=TransitionId::Configure,
       MsgEnable=TransitionId::Enable,
       MsgDisable=TransitionId::Disable, };

namespace Pds {
  namespace Xpm {

#define Q(a,b)      a ## b
#define PV(name)    Q(name, PV)

#define TOU(value)  *reinterpret_cast<unsigned*>(value)
    //#define PRT(value)  printf("%60.60s: %32.32s: 0x%02x\n", __PRETTY_FUNCTION__, _channel.epicsName(), value)
#define PRT(value)  {}

#define PVG(i) {                                        \
      _ctrl.sem().take();                               \
      try {                                                 \
        _ctrl.setPartition();                               \
        PRT(getScalarAs<unsigned>());    _ctrl.module().i;  \
      } catch (CPSWError&) {}                               \
      _ctrl.sem().give(); }
#define PVP(i) {                                        \
      _ctrl.sem().take();                               \
      try {                                             \
        _ctrl.setPartition();                           \
        unsigned v = _ctrl.module().i;                  \
        putFrom<unsigned>(v);                           \
      } catch (CPSWError&) {}                           \
      _ctrl.sem().give();                               \
      PRT( ( getScalarAs<unsigned>() ) );               \
      put(); }

#define CPV(name, updatedBody, connectedBody)                           \
                                                                        \
    class PV(name) : public PVBase                                      \
    {                                                                   \
    public:                                                             \
      PV(name)(PVPCtrls&   ctrl,                                        \
               const char* pvName,                                      \
               unsigned    idx = 0) :                                   \
        PVBase(pvName),                                                 \
        _ctrl(ctrl),                                                    \
        _idx(idx) {}                                                    \
      virtual ~PV(name)() {}                                            \
    public:                                                             \
      void updated();                                                   \
      void onConnect();                                                 \
    public:                                                             \
      void put() { if (this->EpicsPVA::connected())  _channel.put(); }  \
    private:                                                            \
      PVPCtrls& _ctrl;                                                  \
      unsigned  _idx;                                                   \
    };                                                                  \
    void PV(name)::updated()                                            \
    {                                                                   \
      if (_ctrl.enabled())                                              \
        updatedBody                                                     \
    }                                                                   \
    void PV(name)::onConnect()                                          \
    {                                                                   \
      connectedBody                                                     \
    }

    //    CPV(Inhibit,        { PVG(setInhibit  (getScalarAs<unsigned>()));             },
    //                        { PVP(getInhibit  ());                        })
    CPV(XPM,                 {} _ctrl.enable(getScalarAs<unsigned>());,
                             {                                           })
    CPV(TagStream,      { PVG(setTagStream(getScalarAs<unsigned>()));             },
                        { PVP(getTagStream());                        })

    CPV(L0Select,            { _ctrl.l0Select(getScalarAs<unsigned>());
                               _ctrl.setL0Select();           },
                             {                                })
    CPV(L0Select_FixedRate,  { _ctrl.fixedRate(getScalarAs<unsigned>());
                               _ctrl.setL0Select();           },
                             {                                })
    CPV(L0Select_ACRate,     { _ctrl.acRate(getScalarAs<unsigned>());
                               _ctrl.setL0Select();           },
                             {                                })
    CPV(L0Select_ACTimeslot, { _ctrl.acTimeslot(getScalarAs<unsigned>());
                               _ctrl.setL0Select();           },
                             {                                })
    CPV(L0Select_Sequence,   { _ctrl.seqIdx(getScalarAs<unsigned>());
                               _ctrl.setL0Select();           },
                             {                                })
    CPV(L0Select_SeqBit,     { _ctrl.seqBit(getScalarAs<unsigned>());
                               _ctrl.setL0Select();           },
                             {                                })
    CPV(DstSelect,           { _ctrl.dstSelect(getScalarAs<unsigned>());
                               _ctrl.setL0Select();           },
                             {                                })
    CPV(DstSelect_Mask,      { _ctrl.dstMask(getScalarAs<unsigned>());
                               _ctrl.setL0Select();           },
                             {                                })

    CPV(ResetL0     ,   { PVG(resetL0());                         },
                        { PVP(l0Reset ());                        })
    CPV(Run,            { PVG(setL0Enabled (getScalarAs<unsigned>() != 0));   },
                        { PVP(getL0Enabled ());                   })

    CPV(L0Delay,        { PVG(setL0Delay(getScalarAs<unsigned>()));           },
                        { PVP(getL0Delay());                      })
    CPV(L1TrgClear,     { PVG(setL1TrgClr  (getScalarAs<unsigned>()));        },
                        { PVP(getL1TrgClr  ());                   })
    CPV(L1TrgEnable,    { PVG(setL1TrgEnb  (getScalarAs<unsigned>()));        },
                        { PVP(getL1TrgEnb  ());                   })
    CPV(L1TrgSource,    { PVG(setL1TrgSrc  (getScalarAs<unsigned>()));        },
                        { PVP(getL1TrgSrc  ());                   })
    CPV(L1TrgWord,      { PVG(setL1TrgWord (getScalarAs<unsigned>()));        },
                        { PVP(getL1TrgWord ());                   })
    CPV(L1TrgWrite,     { PVG(setL1TrgWrite(getScalarAs<unsigned>()));        },
                        { PVP(getL1TrgWrite());                   })

    CPV(AnaTagReset,    { PVG(_analysisRst = getScalarAs<unsigned>());        },
                        { PVP(_analysisRst);                      })
    CPV(AnaTag,         { PVG(_analysisTag = getScalarAs<unsigned>());        },
                        { PVP(_analysisTag);                      })
    CPV(AnaTagPush,     { PVG(_analysisPush = getScalarAs<unsigned>());       },
                        { PVP(_analysisPush);                     })

    CPV(MsgHeader,      { PVG(messageHdr(getScalarAs<unsigned>()));           },
                        {                                                     })
    CPV(MsgInsert,      { if (getScalarAs<unsigned>()!=0) PVG(messageInsert()); },
                        {                                                     })
    CPV(MsgPayload,     { PVG(messagePayload(getScalarAs<unsigned>()));       },
                        {                                                     })
    CPV(MsgConfigKey,   { _ctrl.configKey     (getScalarAs<unsigned>());      },
                        { _ctrl.configKey     (getScalarAs<unsigned>());      })
    CPV(MsgConfig,      { if (getScalarAs<unsigned>()!=0) _ctrl.msg_config(); },
                        {                                         })
    CPV(MsgEnable,      { if (getScalarAs<unsigned>()!=0) _ctrl.msg_enable(); },
                        {                                         })
    CPV(MsgDisable,     { if (getScalarAs<unsigned>()!=0) _ctrl.msg_disable();},
                        {                                         })
    CPV(MsgClear,       { if (getScalarAs<unsigned>()!=0) _ctrl.msg_clear();  },
                        {                                         })
    CPV(InhInterval,    { PVG(inhibitInt(_idx, getScalarAs<unsigned>()));     },
                        { PVP(inhibitInt(_idx));                  })
    CPV(InhLimit,       { PVG(inhibitLim(_idx, getScalarAs<unsigned>()));     },
                        { PVP(inhibitLim(_idx));                  })
    CPV(InhEnable,      { PVG(inhibitEnb(_idx, getScalarAs<unsigned>()));     },
                        { PVP(inhibitEnb(_idx));                  })

    PVPCtrls::PVPCtrls(Module&  m,
                       Semaphore& sem,
                       PVPStats&  stats,
                       unsigned shelf,
                       unsigned partition) : 
    _pv(0), _m(m), _sem(sem), _stats(stats),
      _shelf(shelf), _partition(partition), _enabled(false),
      _l0Select(0), _dstSelect(0), _dstMask(0), _cfgKey(-1) {}
    PVPCtrls::~PVPCtrls() {}

    void PVPCtrls::allocate(const std::string& title)
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

      //      NPV ( Inhibit             );  // Not implemented in firmware

      //  These PVs must not be updated when XPM is updated
      NPV ( XPM                 );
      NPV ( ResetL0             );
      NPV ( Run                 );
      NPV ( MsgInsert           );
      NPV ( MsgConfig           );
      NPV ( MsgEnable           );
      NPV ( MsgDisable          );
      NPV ( MsgClear            );

      NPV ( TagStream           );
      NPV ( L0Select            );
      NPV ( L0Select_FixedRate  );
      NPV ( L0Select_ACRate     );
      NPV ( L0Select_ACTimeslot );
      NPV ( L0Select_Sequence   );
      NPV ( L0Select_SeqBit     );
      NPV ( DstSelect           );
      NPV ( DstSelect_Mask      );
      NPV ( L0Delay             );
      NPV ( L1TrgClear          );
      NPV ( L1TrgEnable         );
      NPV ( L1TrgSource         );
      NPV ( L1TrgWord           );
      NPV ( L1TrgWrite          );
      NPV ( AnaTagReset         );
      NPV ( AnaTag              );
      NPV ( AnaTagPush          );
      NPV ( MsgHeader           );
      NPV ( MsgPayload          );
      NPV ( MsgConfigKey        );
      NPVN( InhInterval         ,4);
      NPVN( InhLimit            ,4);
      NPVN( InhEnable           ,4);
    }

    void PVPCtrls::enable(unsigned shelf)
    {
      //      printf("PVPCtrls::enable %u(%u)\n", shelf, _shelf);
      bool v = (shelf==_shelf);
      _enabled = v;

      sem().take();
      try {
        setPartition();
        _m.resetL0();
        _m.master(v);
        if (!v)
          _m.setL0Enabled(false);
      } catch (CPSWError&) {}
      sem().give();

      if (v) {
        //  Must not update "XPM" else we call ourselves recursively; and others generate messages
        for(unsigned i=8; i<_pv.size(); i++)
          _pv[i]->updated();

#define PrV(v) {}
        PrV(_l0Select);
        PrV(_fixedRate);
        PrV(_acRate);
        PrV(_acTimeslot);
        PrV(_seqIdx);
        PrV(_seqBit);
#undef PrV
      }
    }

    bool PVPCtrls::enabled() const { return _enabled; }

    Module& PVPCtrls::module() { return _m; }
    Semaphore& PVPCtrls::sem() { return _sem; }

    void PVPCtrls::setPartition() { _m.setPartition(_partition); }

    void PVPCtrls::l0Select  (unsigned v) { _l0Select   = v; }
    void PVPCtrls::fixedRate (unsigned v) { _fixedRate  = v; }
    void PVPCtrls::acRate    (unsigned v) { _acRate     = v; }
    void PVPCtrls::acTimeslot(unsigned v) { _acTimeslot = v; }
    void PVPCtrls::seqIdx    (unsigned v) { _seqIdx     = v; }
    void PVPCtrls::seqBit    (unsigned v) { _seqBit     = v; }
    void PVPCtrls::dstSelect (unsigned v) { _dstSelect  = v; }
    void PVPCtrls::dstMask   (unsigned v) { _dstMask    = v; }
    void PVPCtrls::configKey (unsigned v) { _cfgKey     = v; }

    void PVPCtrls::setL0Select()
    {
      _sem.take();
      try {
        setPartition();
        switch(_l0Select)
          {
          case FixedRate:
            _m.setL0Select_FixedRate(_fixedRate);
            break;
          case ACRate:
            _m.setL0Select_ACRate(_acRate, _acTimeslot);
            break;
          case Sequence:
            _m.setL0Select_Sequence(_seqIdx, _seqBit);
            break;
          default:
            printf("%s: Invalid L0Select value: %d\n", __func__, _l0Select);
            break;
          }
        _m.setL0Select_Destn(_dstSelect,_dstMask);
      } catch (CPSWError&) {}
      _sem.give();
      dump();
    }

    void PVPCtrls::msg_config()
    {
      printf("msg_config [%x]\n",_cfgKey);
      _sem.take();
      try {
        setPartition();                           \
        _m.messagePayload(_cfgKey);
        _m.messageHdr    (TransitionId::Configure);
        _m.messageInsert ();
      } catch (CPSWError&) {}
      _sem.give();
    }

    void PVPCtrls::msg_enable()
    {
      printf("msg_enable\n");
      _sem.take();
      try {
        setPartition();                           \
        _m.messageHdr    (TransitionId::Enable);
        _m.messageInsert ();
      } catch (CPSWError&) {}
      _sem.give();
    }

    void PVPCtrls::msg_disable()
    {
      printf("msg_disable\n");
      _sem.take();
      try {
        setPartition();                           \
        _m.messageHdr    (TransitionId::Disable);
        _m.messageInsert ();
      } catch (CPSWError&) {}
      _sem.give();
    }

    void PVPCtrls::msg_clear()
    {
      printf("msg_clear\n");
      _sem.take();
      try {
        setPartition();                           \
        _m.messagePayload(0);
        _m.messageHdr    (MsgClear);
        _m.messageInsert ();
      } catch (CPSWError&) {}
      _sem.give();
    }

    void PVPCtrls::dump() const
    {
#ifdef DBUG
      printf("partn: %u  l0Select %x\n"
             "\tfR %x  acR %x  acTS %x  seqIdx %x  seqB %x\n"
             "\tdstSel %x  dstMask %x\n",
             _partition, _l0Select, 
             _fixedRate, _acRate, _acTimeslot, _seqIdx, _seqBit,
             _dstSelect, _dstMask);
#endif
    }
  };
};
