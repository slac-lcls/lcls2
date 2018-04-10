#include "psdaq/xpm/PVPCtrls.hh"
#include "psdaq/xpm/Module.hh"
#include "psdaq/xpm/PVPStats.hh"
#include "psdaq/epicstools/PVBase.hh"
#include "psdaq/service/Semaphore.hh"

#include <string>
#include <sstream>

#include <stdio.h>

#define DBUG

using Pds_Epics::PVBase;

//  XPM Message Enum
enum { MsgClear =0,
       MsgDelay =1,
       MsgConfig=2,
       MsgEnable=3,
       MsgDisable=4, };

namespace Pds {
  namespace Xpm {

#define Q(a,b)      a ## b
#define PV(name)    Q(name, PV)

#define TOU(value)  *reinterpret_cast<unsigned*>(value)
    //#define PRT(value)  printf("%60.60s: %32.32s: 0x%02x\n", __PRETTY_FUNCTION__, _channel.epicsName(), value)
#define PRT(value)  {}

#define PVG(i) {                                \
      _ctrl.sem().take();                       \
      _ctrl.setPartition();                     \
      PRT(TOU(data()));    _ctrl.module().i;    \
      _ctrl.sem().give(); }
#define PVP(i) {                                        \
      _ctrl.sem().take();                               \
      _ctrl.setPartition();                             \
      TOU(data()) = _ctrl.module().i;                   \
      _ctrl.sem().give();                               \
      PRT( ( TOU(data()) ) );                           \
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
      void connected(bool);                                             \
    public:                                                             \
      void put() { if (this->EpicsCA::connected())  _channel.put(); }   \
    private:                                                            \
      PVPCtrls& _ctrl;                                                  \
      unsigned  _idx;                                                   \
    };                                                                  \
    void PV(name)::updated()                                            \
    {                                                                   \
      if (_ctrl.enabled())                                              \
        updatedBody                                                     \
    }                                                                   \
    void PV(name)::connected(bool c)                                    \
    {                                                                   \
      this->EpicsCA::connected(c);                                      \
      connectedBody                                                     \
    }

    //    CPV(Inhibit,        { PVG(setInhibit  (TOU(data())));             },
    //                        { PVP(getInhibit  ());                        })
    CPV(XPM,                 {} _ctrl.enable(TOU(data()));,
                             {                                           })
    CPV(TagStream,      { PVG(setTagStream(TOU(data())));             },
                        { PVP(getTagStream());                        })

    CPV(L0Select,            { _ctrl.l0Select(TOU(data()));
                               _ctrl.setL0Select();           },
                             {                                })
    CPV(L0Select_FixedRate,  { _ctrl.fixedRate(TOU(data()));
                               _ctrl.setL0Select();           },
                             {                                })
    CPV(L0Select_ACRate,     { _ctrl.acRate(TOU(data()));
                               _ctrl.setL0Select();           },
                             {                                })
    CPV(L0Select_ACTimeslot, { _ctrl.acTimeslot(TOU(data()));
                               _ctrl.setL0Select();           },
                             {                                })
    CPV(L0Select_Sequence,   { _ctrl.seqIdx(TOU(data()));
                               _ctrl.setL0Select();           },
                             {                                })
    CPV(L0Select_SeqBit,     { _ctrl.seqBit(TOU(data()));
                               _ctrl.setL0Select();           },
                             {                                })
    CPV(DstSelect,           { _ctrl.dstSelect(TOU(data()));
                               _ctrl.setL0Select();           },
                             {                                })
    CPV(DstSelect_Mask,      { _ctrl.dstMask(TOU(data()));
                               _ctrl.setL0Select();           },
                             {                                })

    CPV(ResetL0     ,   { PVG(resetL0());                         },
                        { PVP(l0Reset ());                        })
    CPV(Run,            { PVG(setL0Enabled (TOU(data()) != 0));   },
                        { PVP(getL0Enabled ());                   })

    CPV(L0Delay,        { PVG(setL0Delay(_idx, TOU(data())));     },
                        { PVP(getL0Delay(_idx));                  })
    CPV(L1TrgClear,     { PVG(setL1TrgClr  (_idx, TOU(data())));  },
                        { PVP(getL1TrgClr  (_idx));               })
    CPV(L1TrgEnable,    { PVG(setL1TrgEnb  (_idx, TOU(data())));  },
                        { PVP(getL1TrgEnb  (_idx));               })
    CPV(L1TrgSource,    { PVG(setL1TrgSrc  (_idx, TOU(data())));  },
                        { PVP(getL1TrgSrc  (_idx));               })
    CPV(L1TrgWord,      { PVG(setL1TrgWord (_idx, TOU(data())));  },
                        { PVP(getL1TrgWord (_idx));               })
    CPV(L1TrgWrite,     { PVG(setL1TrgWrite(_idx, TOU(data())));  },
                        { PVP(getL1TrgWrite(_idx));               })

    CPV(AnaTagReset,    { PVG(_analysisRst = TOU(data()));        },
                        { PVP(_analysisRst);                      })
    CPV(AnaTag,         { PVG(_analysisTag = TOU(data()));        },
                        { PVP(_analysisTag);                      })
    CPV(AnaTagPush,     { PVG(_analysisPush = TOU(data()));       },
                        { PVP(_analysisPush);                     })

    CPV(PipelineDepth,  { PVG(_pipelineDepth = TOU(data()));      },
                        { PVP(_pipelineDepth);                    })

    CPV(MsgHeader,      { _ctrl.msgHeader(TOU(data()));           },
                        { _ctrl.msgHeader(TOU(data()));           })
    CPV(MsgInsert,      { if (TOU(data())!=0) _ctrl.msgInsert();  },
                        {                                         })
    CPV(MsgPayload,     { _ctrl.msgPayload(TOU(data()));          },
                        { _ctrl.msgPayload(TOU(data()));          })
    CPV(MsgConfigKey,   { _ctrl.configKey     (TOU(data()));      },
                        { _ctrl.configKey     (TOU(data()));      })
    CPV(MsgConfig,      { if (TOU(data())!=0) _ctrl.msg_config(); },
                        {                                         })
    CPV(MsgEnable,      { if (TOU(data())!=0) _ctrl.msg_enable(); },
                        {                                         })
    CPV(MsgDisable,     { if (TOU(data())!=0) _ctrl.msg_disable();},
                        {                                         })
    CPV(MsgClear,       { if (TOU(data())!=0) _ctrl.msg_clear();  },
                        {                                         })
    CPV(InhInterval,    { PVG(inhibitInt(_idx, TOU(data())));     },
                        { PVP(inhibitInt(_idx));                  })
    CPV(InhLimit,       { PVG(inhibitLim(_idx, TOU(data())));     },
                        { PVP(inhibitLim(_idx));                  })
    CPV(InhEnable,      { PVG(inhibitEnb(_idx, TOU(data())));     },
                        { PVP(inhibitEnb(_idx));                  })

    PVPCtrls::PVPCtrls(Module&  m,
                       Semaphore& sem,
                       PVPStats&  stats,
                       unsigned shelf,
                       unsigned partition) : 
    _pv(0), _m(m), _sem(sem), _stats(stats),
      _shelf(shelf), _partition(partition), _enabled(false) {}
    PVPCtrls::~PVPCtrls() {}

    void PVPCtrls::allocate(const std::string& title)
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

      //      NPV ( Inhibit             );  // Not implemented in firmware
      NPV ( XPM                 );
      NPV ( TagStream           );
      NPV ( L0Select            );
      NPV ( L0Select_FixedRate  );
      NPV ( L0Select_ACRate     );
      NPV ( L0Select_ACTimeslot );
      NPV ( L0Select_Sequence   );
      NPV ( L0Select_SeqBit     );
      NPV ( DstSelect           );
      NPV ( DstSelect_Mask      );
      NPV ( ResetL0             );
      NPV ( Run                 );
      NPV ( L1TrgClear          );
      NPV ( L1TrgEnable         );
      NPV ( L1TrgSource         );
      NPV ( L1TrgWord           );
      NPV ( L1TrgWrite          );
      NPV ( AnaTagReset         );
      NPV ( AnaTag              );
      NPV ( AnaTagPush          );
      NPV ( PipelineDepth       );
      NPV ( MsgHeader           );
      NPV ( MsgInsert           );
      NPV ( MsgPayload          );
      NPV ( MsgConfig           );
      NPV ( MsgConfigKey        );
      NPV ( MsgEnable           );
      NPV ( MsgDisable          );
      NPV ( MsgClear            );
      NPVN( InhInterval         ,4);
      NPVN( InhLimit            ,4);
      NPVN( InhEnable           ,4);

      // Wait for monitors to be established
      ca_pend_io(0);
    }

    void PVPCtrls::enable(unsigned shelf)
    {
      printf("PVPCtrls::enable %u(%u)\n", shelf, _shelf);
      bool v = (shelf==_shelf);
      _enabled = v;

      if (v) {
        sem().take();
        setPartition();
        _m.resetL0();
        sem().give();

        //  Must not update "XPM" else we call ourselves recursively
        for(unsigned i=1; i<_pv.size(); i++)
          _pv[i]->updated();

        //#define PrV(v) printf("\t %15.15s: %x\n", #v, v)
#define PrV(v) {}
        PrV(_l0Select);
        PrV(_fixedRate);
        PrV(_acRate);
        PrV(_acTimeslot);
        PrV(_seqIdx);
        PrV(_seqBit);
#undef PrV
      }
      if (!v)
        _m.setL0Enabled(false);
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
    void PVPCtrls::msgHeader (unsigned v) { _msgHdr     = v; }
    void PVPCtrls::msgPayload(unsigned v) { _msgPayload = v; }
    void PVPCtrls::configKey (unsigned v) { _cfgKey     = v; }

    void PVPCtrls::setL0Select()
    {
      _sem.take();
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
      _sem.give();
      dump();
    }

    void PVPCtrls::msgInsert()
    {
      printf("msg_insert [%x]\n", _msgHdr);
      _sem.take();
      _m.messagePayload(_partition, _msgPayload);
      _m.messageHdr    (_partition, _msgHdr);
      _sem.give();
    }

    void PVPCtrls::msg_config()
    {
      printf("msg_config [%x]\n",_cfgKey);
      _sem.take();
      _m.messagePayload(_partition, _cfgKey);
      _m.messageHdr    (_partition, MsgConfig);
      _sem.give();
    }

    void PVPCtrls::msg_enable()
    {
      printf("msg_enable\n");
      _sem.take();
      _m.messageHdr    (_partition, MsgEnable);
      _sem.give();
    }

    void PVPCtrls::msg_disable()
    {
      printf("msg_disable\n");
      _sem.take();
      _m.messageHdr    (_partition, MsgDisable);
      _sem.give();
    }

    void PVPCtrls::msg_clear()
    {
      printf("msg_clear\n");
      _sem.take();
      _m.messagePayload(_partition, 0);
      _m.messageHdr    (_partition, MsgClear);
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
