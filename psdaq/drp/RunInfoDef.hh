#ifndef Drp_RunInfoDef_hh
#define Drp_RunInfoDef_hh

#include "xtcdata/xtc/NamesId.hh"
#include "xtcdata/xtc/VarDef.hh"

namespace Drp
{

class RunInfoDef : public XtcData::VarDef
{
public:
  enum index
    {
        EXPT,
        RUNNUM
    };

  RunInfoDef()
   {
       XtcData::VarDef::NameVec.push_back({"expt", XtcData::Name::CHARSTR,1});
       XtcData::VarDef::NameVec.push_back({"runnum", XtcData::Name::UINT32});
   }
};

}
#endif
