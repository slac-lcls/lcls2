#include "pdsdata/xtc/BldInfo.hh"
#include "pdsdata/xtc/Level.hh"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

using namespace Pds;

BldInfo::BldInfo(uint32_t processId, Type type) : Src(Level::Reporter) {
  _log |= processId&0x00ffffff;
  _phy = type;
}

BldInfo::BldInfo(const char* sname) : Src(Level::Reporter)
{
  for(unsigned i=0; i<NumberOf; i++) {
    _phy = i;
    const char* bname = name(*this);
    unsigned len = strlen(bname);
    if (strncmp(sname,bname,len)==0) {
      return;
    }
  }
  _phy = NumberOf;
}

bool BldInfo::operator==(const BldInfo& o) const
{
  return o.phy()==_phy;
}

uint32_t BldInfo::processId() const { return _log&0xffffff; }

BldInfo::Type BldInfo::type() const {return (BldInfo::Type)(_phy); }

const char* BldInfo::name(const BldInfo& src){
  static const char* _typeNames[] = {
    "EBeam",
    "PhaseCavity",
    "FEEGasDetEnergy",
    "NH2-SB1-IPM-01",
    "XCS-IPM-01",
    "XCS-DIO-01",
    "XCS-IPM-02",
    "XCS-DIO-02",
    "XCS-IPM-03",
    "XCS-DIO-03",
    "XCS-IPM-03m",
    "XCS-DIO-03m",
    "XCS-YAG-1",
    "XCS-YAG-2",
    "XCS-YAG-3m",
    "XCS-YAG-3",
    "XCS-YAG-mono",
    "XCS-IPM-mono",
    "XCS-DIO-mono",
    "XCS-DEC-mono",
    "MEC-LAS-EM-01",
    "MEC-TCTR-PIP-01",
    "MEC-TCTR-DI-01",
    "MEC-XT2-IPM-02",
    "MEC-XT2-IPM-03",
    "MEC-HXM-IPM-01",
    "GMD",
    "CxiDg1_Imb01",
    "CxiDg2_Imb01",
    "CxiDg2_Imb02",
    "CxiDg3_Imb01",
    "CxiDg1_Pim",
    "CxiDg2_Pim",
    "CxiDg3_Pim",
    "XppMon_Pim0",
    "XppMon_Pim1",
    "XppSb2_Ipm",
    "XppSb3_Ipm",
    "XppSb3_Pim",
    "XppSb4_Pim",
    "XppEnds_Ipm0",
    "XppEnds_Ipm1",
    "MEC-XT2-PIM-02",
    "MEC-XT2-PIM-03",
    "CxiDg3_Spec",
    "NH2-SB1-IPM-02",
    "FEE-SPEC0",
    "SXR-SPEC0",
    "XPP-SPEC0",
    "XCS-USR-IPM-01",
    "XCS-USR-IPM-02",
    "XCS-USR-IPM-03",
    "XCS-USR-IPM-04",
    "XCS-IPM-04",
    "XCS-DIO-04",
    "XCS-IPM-05",
    "XCS-DIO-05",
    "XCS-IPM-gon",
    "XCS-IPM-ladm",
    "XPP-AIN-01",
    "XCS-AIN-01",
    "AMO-AIN-01",
    "MFX-BEAMMON-01",
    "EOrbits",
    "MfxDg1_Pim",
    "MfxDg2_Pim",
    "SXR-AIN-01",
    "HX2-BEAMMON-01"
  };
  return (src.type() < NumberOf ? _typeNames[src.type()] : "-Invalid-");
}
