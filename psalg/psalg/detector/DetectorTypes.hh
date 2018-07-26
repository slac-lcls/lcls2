#ifndef PSALG_DETECTORTYPES_H
#define PSALG_DETECTORTYPES_H

/** Usage
 *
 * #include "psalg/calib/DetectorTypes.hh"
 */

#include <string>
#include <map>

//-------------------

namespace detector {

  enum DETTYPE {NONDEFINED_DETECTOR=0,
                AREA_DETECTOR,
                CONTROL_DATA_DETECTOR,
		DDL_DETECTOR,
                EPICS_DETECTOR,
                EVR_DETECTOR,
                WF_DETECTOR,
                IPIMB_DETECTOR,
                OCEAN_DETECTOR,
                USDUSB_DETECTOR,
                TDC_DETECTOR
               };

  static std::map<std::string, DETTYPE> map_detname_to_dettype = {
    {"NoDevice"    , NONDEFINED_DETECTOR},
    {"Evr"         , EVR_DETECTOR},
    {"Acqiris"     , WF_DETECTOR},
    {"Opal1000"    , AREA_DETECTOR},
    {"Tm6740"      , AREA_DETECTOR},
    {"pnCCD"       , AREA_DETECTOR},
    {"Princeton"   , AREA_DETECTOR},
    {"Fccd"        , AREA_DETECTOR},
    {"Ipimb"       , IPIMB_DETECTOR},
    {"Encoder"     , DDL_DETECTOR},
    {"Cspad"       , AREA_DETECTOR},
    {"AcqTDC"      , TDC_DETECTOR},
    {"Xamps"       , AREA_DETECTOR},
    {"Cspad2x2"    , AREA_DETECTOR},
    {"Fexamp"      , AREA_DETECTOR},
    {"Gsc16ai"     , DDL_DETECTOR},
    {"Phasics"     , AREA_DETECTOR},
    {"Timepix"     , AREA_DETECTOR},
    {"Opal2000"    , AREA_DETECTOR},
    {"Opal4000"    , AREA_DETECTOR},
    {"OceanOptics" , OCEAN_DETECTOR},
    {"Opal1600"    , AREA_DETECTOR},
    {"Opal8000"    , AREA_DETECTOR},
    {"Fli"         , AREA_DETECTOR},
    {"Quartz4A150" , AREA_DETECTOR},
    {"DualAndor"   , AREA_DETECTOR},
    {"Andor"       , AREA_DETECTOR},
    {"USDUSB"      , USDUSB_DETECTOR},
    {"OrcaFl40"    , AREA_DETECTOR},
    {"Imp"         , WF_DETECTOR},
    {"Epix"        , AREA_DETECTOR},
    {"Rayonix"     , AREA_DETECTOR},
    {"EpixSampler" , AREA_DETECTOR},
    {"Pimax"       , AREA_DETECTOR},
    {"Fccd960"     , AREA_DETECTOR},
    {"Epix10k"     , AREA_DETECTOR},
    {"Epix10ka"    , AREA_DETECTOR},
    {"Epix100a"    , AREA_DETECTOR},
    {"EpixS"       , AREA_DETECTOR},
    {"Gotthard"    , AREA_DETECTOR},
    {"Wave8"       , WF_DETECTOR},
    {"LeCroy"      , WF_DETECTOR},
    {"Archon"      , AREA_DETECTOR},
    {"Jungfrau"    , AREA_DETECTOR},
    {"Zyla"        , AREA_DETECTOR},    
    {"ControlsCamera" , AREA_DETECTOR}
  };

  static std::map<std::string, DETTYPE> map_bldinfo_to_dettype = {
    {"EBeam"              , DDL_DETECTOR},
    {"PhaseCavity"        , DDL_DETECTOR},
    {"FEEGasDetEnergy"    , DDL_DETECTOR},
    {"NH2-SB1-IPM-01"     , IPIMB_DETECTOR},
    {"XCS-IPM-01"         , IPIMB_DETECTOR},
    {"XCS-DIO-01"         , IPIMB_DETECTOR},
    {"XCS-IPM-02"         , IPIMB_DETECTOR},
    {"XCS-DIO-02"         , IPIMB_DETECTOR},
    {"XCS-IPM-03"         , IPIMB_DETECTOR},
    {"XCS-DIO-03"         , IPIMB_DETECTOR},
    {"XCS-IPM-03m"        , IPIMB_DETECTOR},
    {"XCS-DIO-03m"        , IPIMB_DETECTOR},
    {"XCS-YAG-1"          , DDL_DETECTOR},
    {"XCS-YAG-2"          , DDL_DETECTOR},
    {"XCS-YAG-3m"         , DDL_DETECTOR},
    {"XCS-YAG-3"          , DDL_DETECTOR},
    {"XCS-YAG-mono"       , DDL_DETECTOR},
    {"XCS-IPM-mono"       , IPIMB_DETECTOR},
    {"XCS-DIO-mono"       , IPIMB_DETECTOR},
    {"XCS-DEC-mono"       , DDL_DETECTOR},
    {"MEC-LAS-EM-01"      , IPIMB_DETECTOR},
    {"MEC-TCTR-PIP-01"    , IPIMB_DETECTOR},
    {"MEC-TCTR-DI-01"     , IPIMB_DETECTOR},
    {"MEC-XT2-IPM-02"     , IPIMB_DETECTOR},
    {"MEC-XT2-IPM-03"     , IPIMB_DETECTOR},
    {"MEC-HXM-IPM-01"     , IPIMB_DETECTOR},
    {"GMD"                , DDL_DETECTOR},
    {"CxiDg1_Imb01"       , IPIMB_DETECTOR},
    {"CxiDg2_Imb01"       , IPIMB_DETECTOR},
    {"CxiDg2_Imb02"       , IPIMB_DETECTOR},
    {"CxiDg3_Imb01"       , IPIMB_DETECTOR},
    {"CxiDg1_Pim"         , IPIMB_DETECTOR},
    {"CxiDg2_Pim"         , IPIMB_DETECTOR},
    {"CxiDg3_Pim"         , IPIMB_DETECTOR},
    {"XppMon_Pim0"        , IPIMB_DETECTOR},
    {"XppMon_Pim1"        , IPIMB_DETECTOR},
    {"XppSb2_Ipm"         , IPIMB_DETECTOR},
    {"XppSb3_Ipm"         , IPIMB_DETECTOR},
    {"XppSb3_Pim"         , IPIMB_DETECTOR},
    {"XppSb4_Pim"         , IPIMB_DETECTOR},
    {"XppEnds_Ipm0"       , IPIMB_DETECTOR},
    {"XppEnds_Ipm1"       , IPIMB_DETECTOR},
    {"MEC-XT2-PIM-02"     , IPIMB_DETECTOR},
    {"MEC-XT2-PIM-03"     , IPIMB_DETECTOR},
    {"CxiDg3_Spec"        , DDL_DETECTOR},
    {"NH2-SB1-IPM-02"     , IPIMB_DETECTOR},
    {"FEE-SPEC0"          , DDL_DETECTOR},
    {"SXR-SPEC0"          , DDL_DETECTOR},
    {"XPP-SPEC0"          , DDL_DETECTOR},
    {"XCS-USR-IPM-01"     , IPIMB_DETECTOR},
    {"XCS-USR-IPM-02"     , IPIMB_DETECTOR},
    {"XCS-USR-IPM-03"     , IPIMB_DETECTOR},
    {"XCS-USR-IPM-04"     , IPIMB_DETECTOR},
    {"XCS-IPM-04"         , IPIMB_DETECTOR},
    {"XCS-DIO-04"         , IPIMB_DETECTOR},
    {"XCS-IPM-05"         , IPIMB_DETECTOR},
    {"XCS-DIO-05"         , IPIMB_DETECTOR},
    {"XCS-IPM-gon"        , IPIMB_DETECTOR},
    {"XCS-IPM-ladm"       , IPIMB_DETECTOR},
    {"XPP-AIN-01"         , DDL_DETECTOR},
    {"XCS-AIN-01"         , DDL_DETECTOR},
    {"AMO-AIN-01"         , DDL_DETECTOR},
    {"MFX-BEAMMON-01"     , DDL_DETECTOR},
    {"EOrbits"            , DDL_DETECTOR},
    {"MfxDg1_Pim"         , DDL_DETECTOR},
    {"MfxDg2_Pim"         , DDL_DETECTOR},
    {"SXR-AIN-01"         , DDL_DETECTOR},
    {"HX2-SB1-BMMON"      , DDL_DETECTOR},
    {"XRT-USB-ENCODER-01" , USDUSB_DETECTOR},
    {"XPP-USB-ENCODER-01" , USDUSB_DETECTOR},
    {"XPP-USB-ENCODER-02" , USDUSB_DETECTOR},
    {"XCS-USB-ENCODER-01" , USDUSB_DETECTOR},
    {"CXI-USB-ENCODER-01" , USDUSB_DETECTOR},
    {"XCS-SND-DIO"        , DDL_DETECTOR},
    {"MFX-USR-DIO"        , DDL_DETECTOR},
    {"XPP-SB2-BMMON"      , DDL_DETECTOR},
    {"XPP-SB3-BMMON"      , DDL_DETECTOR},
    {"HFX-DG2-BMMON"      , DDL_DETECTOR},
    {"XCS-SB1-BMMON"      , DDL_DETECTOR},
    {"XCS-SB2-BMMON"      , DDL_DETECTOR},
    {"CXI-DG2-BMMON"      , DDL_DETECTOR},
    {"CXI-DG3-BMMON"      , DDL_DETECTOR},
    {"MFX-DG1-BMMON"      , DDL_DETECTOR},
    {"MFX-DG2-BMMON"      , DDL_DETECTOR},
    {"MFX-AIN-01"         , DDL_DETECTOR},
    {"MEC-AIN-01"         , DDL_DETECTOR},
    {"FEE-AIN-01"         , DDL_DETECTOR}
  };

  const DETTYPE find_dettype(const std::string& detname);
  void print_map_detname_to_dettype();
  void print_map_bldinfo_to_dettype();

} // namespace detector

//-------------------

#endif // PSALG_DETECTORTYPES_H

