#ifndef EPICS_XTC_SETTINGS_H
#define EPICS_XTC_SETTINGS_H

#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/VarDef.hh"

namespace Drp
{

namespace EpicsXtcSettings
{
  enum {iRawNamesIndex = 0, iInfoNamesIndex}; // < 255

  const int iMaxNumPv = 10000;

  /*
   * 200 Bytes: For storing a DBR_CTRL_DOUBLE PV
   */
  const int iMaxXtcSize = iMaxNumPv * 200;
}

class EpicsArchDef : public XtcData::VarDef
{
public:
  enum index
  {
    Stale,
    Data,                               // Pseudonym for the start of PV data
  };

  EpicsArchDef()
  {
    const size_t rank = 1;
    NameVec.push_back({"StaleFlags", XtcData::Name::UINT32, rank});
    // PVs are added to NameVec in the code
  }
};

}

#endif
