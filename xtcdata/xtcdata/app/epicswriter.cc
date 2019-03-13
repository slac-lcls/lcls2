#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/TypeId.hh"
#include "xtcdata/xtc/XtcIterator.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "rapidjson/document.h"
#include "xtcdata/xtc/XtcFileIterator.hh"

#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <unistd.h>
#include <type_traits>
#include <cstring>
#include <sys/time.h>
#include <stdint.h>
#include <fstream>
#include <iostream>
#include <array>
#include <fcntl.h>

using namespace XtcData;
using namespace rapidjson;
using namespace std;
using std::string;

#define BUFSIZE 0x4000000


class EpicsDef:public VarDef
{
public:
  enum index
    {
#ifndef EPICSSTREAMID_UNITTEST
        HX2_DVD_GCC_01_PMON,
        HX2_DVD_GPI_01_PMON,
        HX2_DVD_PIP_01_VMON,
        HX2_SB1_IPM_01_ChargeAmpRangeCH0,
        HX2_SB1_IPM_01_DiodeBias,
        HX2_SB1_IPM_02_ChargeAmpRangeCH0,
        HX2_SB1_IPM_02_DiodeBias,
        HX2_SB1_JAWS_ACTUAL_XCENTER,
        HX2_SB1_JAWS_ACTUAL_XWIDTH,
        HX2_SB1_JAWS_ACTUAL_YCENTER,
        HX2_SB1_JAWS_ACTUAL_YWIDTH,
        HX2_SB1_MMS_02_RBV,
        HX2_SB1_MMS_03_RBV,
        HX2_SB1_MMS_04_RBV,
        HX2_SB1_MMS_05_RBV,
        HX2_SB1_MMS_06_RBV,
        HX2_SB1_MMS_06_RRBV,
        HX2_SB1_MMS_07_RBV,
        HX2_SB1_MMS_07_RRBV,
        HX2_SB1_MMS_08_RBV,
        HX2_SB1_MMS_08_RRBV,
        HX2_UVD_GCC_01_PMON,
        HX2_UVD_GPI_01_PMON,
        HX2_UVD_PIP_01_VMON,
        HX3_DVD_GPI_01_PMON,
        HX3_DVD_PIP_01_VMON,
        HX3_MON_GCC_01_PMON,
        HX3_MON_GCC_02_PMON,
        HX3_MON_GPI_01_PMON,
        HX3_MON_PIP_01_VMON,
        HX3_MON_PIP_02_VMON,
        HX3_MON_PIP_03_VMON,
        LAS_FS3_Angle_Shift_Ramp_Target,
        LAS_FS3_REG_Angle_Shift_rd,
        LAS_FS3_REG_kp_vcxo_rd,
        LAS_FS3_VIT_FS_CTR_TIME,
        LAS_FS3_VIT_FS_TGT_TIME,
        LAS_FS3_WAVE_Signal_T_AVG,
        LAS_FS3_alldiff_fs,
        LAS_FS3_alldiff_fs_RMS,
        LAS_R54_EVR_27_CTRL_DG0C,
        LAS_R54_EVR_27_CTRL_DG0D,
        LAS_R54_EVR_27_CTRL_DG1C,
        LAS_R54_EVR_27_CTRL_DG1D,
        LAS_XPP_DDG_01_aDelayAO,
        PLC_XPP_LSS_C114I,
        PLC_XPP_LSS_C116I,
        PLC_XPP_LSS_C120I,
        PLC_XPP_LSS_C122I,
        PLC_XPP_LSS_C124I,
        PLC_XPP_LSS_C126I,
        ROOM_BSY0_1_OUTSIDETEMP,
        STEP_FEE1_441_MOTR_VAL,
        STEP_FEE1_442_MOTR_VAL,
        STEP_FEE1_443_MOTR_VAL,
        STEP_FEE1_444_MOTR_VAL,
        STEP_FEE1_445_MOTR_VAL,
        STEP_FEE1_446_MOTR_VAL,
        STEP_FEE1_447_MOTR_VAL,
        XPP_ATT_COM_R3_CUR,
        XPP_ATT_COM_R_CUR,
        XPP_ATT_COM_T_CALC_VALE,
        XPP_GON_MMS_01_RBV,
        XPP_GON_MMS_02_RBV,
        XPP_GON_MMS_03_RBV,
        XPP_GON_MMS_04_RBV,
        XPP_GON_MMS_05_RBV,
        XPP_GON_MMS_06_RBV,
        XPP_GON_MMS_07_RBV,
        XPP_GON_MMS_08_RBV,
        XPP_GON_MMS_09_RBV,
        XPP_GON_MMS_10_RBV,
        XPP_GON_MMS_11_RBV,
        XPP_GON_MMS_12_RBV,
        XPP_GON_MMS_13_RBV,
        XPP_GON_MMS_14_RBV,
        XPP_GON_MMS_15_RBV,
        XPP_GON_MMS_16_RBV,
        XPP_IPM1_TARGET_Y_STATE,
        XPP_IPM2_TARGET_Y_STATE,
        XPP_IPM3_TARGET_Y_STATE,
        XPP_LAS_MMN_01_RBV,
        XPP_LAS_MMN_02_RBV,
        XPP_LAS_MMN_03_RBV,
        XPP_LAS_MMN_05_RBV,
        XPP_LAS_MMN_06_RBV,
        XPP_LAS_MMN_08_RBV,
        XPP_LAS_MMN_09_RBV,
        XPP_LAS_MMN_10_RBV,
        XPP_LAS_MMN_11_RBV,
        XPP_LAS_MMN_12_RBV,
        XPP_LAS_MMN_13_RBV,
        XPP_LAS_MMN_14_RBV,
        XPP_LAS_MMN_15_RBV,
        XPP_LAS_MMN_16_RBV,
        XPP_MON_MMS_04_RBV,
        XPP_MON_MMS_05_RBV,
        XPP_MON_MMS_06_RBV,
        XPP_MON_MMS_07_RBV,
        XPP_MON_MMS_07_VAL,
        XPP_MON_MMS_08_ACCL,
        XPP_MON_MMS_08_RBV,
        XPP_MON_MMS_08_VAL,
        XPP_MON_MMS_08_C1,
        XPP_MON_MMS_08_C2,
        XPP_MON_MMS_08_EL,
        XPP_MON_MMS_09_RBV,
        XPP_MON_MMS_10_RBV,
        XPP_MON_MMS_11_RBV,
        XPP_MON_MMS_12_RBV,
        XPP_MON_MMS_13_RBV,
        XPP_MON_MMS_13_VAL,
        XPP_MON_MMS_14_ACCL,
        XPP_MON_MMS_14_RBV,
        XPP_MON_MMS_14_VAL,
        XPP_MON_MMS_14_C1,
        XPP_MON_MMS_14_C2,
        XPP_MON_MMS_14_EL,
        XPP_MON_MMS_15_RBV,
        XPP_MON_MMS_16_RBV,
        XPP_MON_MMS_17_RBV,
        XPP_MON_MMS_18_RBV,
        XPP_MON_MMS_19_RBV,
        XPP_MON_MMS_20_RBV,
        XPP_MON_MMS_22_RBV,
        XPP_MON_MMS_23_RBV,
        XPP_MON_MPZ_07A_POSITIONGET,
        XPP_MON_MPZ_08_POSITIONGET,
        XPP_SB2_GCC_01_PMON,
        XPP_SB2_GPI_01_PMON,
        XPP_SB2_IPM_01_ChargeAmpRangeCH0,
        XPP_SB2_IPM_01_DiodeBias,
        XPP_SB2_MMS_01_RBV,
        XPP_SB2_MMS_01_RRBV,
        XPP_SB2_MMS_02_RBV,
        XPP_SB2_MMS_02_RRBV,
        XPP_SB2_MMS_03_RBV,
        XPP_SB2_MMS_03_RRBV,
        XPP_SB2_MMS_05_RBV,
        XPP_SB2_MMS_06_RBV,
        XPP_SB2_MMS_07_RBV,
        XPP_SB2_MMS_08_RBV,
        XPP_SB2_MMS_09_RBV,
        XPP_SB2_MMS_10_RBV,
        XPP_SB2_MMS_11_RBV,
        XPP_SB2_MMS_12_RBV,
        XPP_SB2_MMS_13_RBV,
        XPP_SB2_MMS_14_RBV,
        XPP_SB2_MMS_15_RBV,
        XPP_SB2_MMS_17_RBV,
        XPP_SB2_MMS_17_RRBV,
        XPP_SB2_MMS_18_RBV,
        XPP_SB2_MMS_18_RRBV,
        XPP_SB2_MMS_19_RBV,
        XPP_SB2_MMS_19_RRBV,
        XPP_SB2_MMS_20_RBV,
        XPP_SB2_MMS_20_RRBV,
        XPP_SB2_MMS_21_RBV,
        XPP_SB2_MMS_21_RRBV,
        XPP_SB2_MMS_22_RBV,
        XPP_SB2_MMS_22_RRBV,
        XPP_SB2_MMS_23_RBV,
        XPP_SB2_MMS_23_RRBV,
        XPP_SB2_MMS_24_RBV,
        XPP_SB2_MMS_24_RRBV,
        XPP_SB2_MMS_25_RBV,
        XPP_SB2_MMS_25_RRBV,
        XPP_SB2_MMS_26_RBV,
        XPP_SB2_MMS_26_RRBV,
        XPP_SB2_MMS_27_RBV,
        XPP_SB2_MMS_27_RRBV,
        XPP_SB2H_JAWS_ACTUAL_XCENTER,
        XPP_SB2H_JAWS_ACTUAL_XWIDTH,
        XPP_SB2H_JAWS_ACTUAL_YCENTER,
        XPP_SB2H_JAWS_ACTUAL_YWIDTH,
        XPP_SB2L_JAWS_ACTUAL_XCENTER,
        XPP_SB2L_JAWS_ACTUAL_XWIDTH,
        XPP_SB2L_JAWS_ACTUAL_YCENTER,
        XPP_SB2L_JAWS_ACTUAL_YWIDTH,
        XPP_SB3_CLF_01_RBV,
        XPP_SB3_CLZ_01_RBV,
        XPP_SB3_GCC_01_PMON,
        XPP_SB3_GCC_02_PMON,
        XPP_SB3_GPI_01_PMON,
        XPP_SB3_GPI_02_PMON,
        XPP_SB3_IPM_01_ChargeAmpRangeCH0,
        XPP_SB3_IPM_01_DiodeBias,
        XPP_SB3_JAWS_ACTUAL_XCENTER,
        XPP_SB3_JAWS_ACTUAL_XWIDTH,
        XPP_SB3_JAWS_ACTUAL_YCENTER,
        XPP_SB3_JAWS_ACTUAL_YWIDTH,
        XPP_SB3_MMS_01_RBV,
        XPP_SB3_MMS_02_RBV,
        XPP_SB3_MMS_03_RBV,
        XPP_SB3_MMS_04_RBV,
        XPP_SB3_MMS_05_RBV,
        XPP_SB3_MMS_06_RBV,
        XPP_SB3_MMS_07_RBV,
        XPP_SB3_MMS_08_RBV,
        XPP_SB3_MMS_09_RBV,
        XPP_SB3_MMS_10_RBV,
        XPP_SB3_MMS_11_RBV,
        XPP_SB3_MMS_11_RRBV,
        XPP_SB3_MMS_12_RBV,
        XPP_SB3_MMS_12_RRBV,
        XPP_SB3_MMS_13_RBV,
        XPP_SB3_MMS_13_RRBV,
        XPP_SB3_MMS_14_RBV,
        XPP_SB3_MMS_14_RRBV,
        XPP_SB3_MMS_15_RBV,
        XPP_SB3_MMS_15_RRBV,
        XPP_SB3_PIP_01_VMON,
        XPP_SB4_GCC_01_PMON,
        XPP_SB4_GPI_01_PMON,
        XPP_SB4_IPM_01_ChargeAmpRangeCH0,
        XPP_SB4_IPM_01_ChargeAmpRangeCH1,
        XPP_SB4_IPM_01_ChargeAmpRangeCH2,
        XPP_SB4_IPM_01_ChargeAmpRangeCH3,
        XPP_SB4_IPM_01_DiodeBias,
        XPP_SB4_USR_MMS_42,
        XPP_SB4_USR_MMS_43,
        XPP_SCAN_ISSCAN,
        XPP_SCAN_ISTEP,
        XPP_SCAN_MAX00,
        XPP_SCAN_MAX01,
        XPP_SCAN_MAX02,
        XPP_SCAN_MIN00,
        XPP_SCAN_MIN01,
        XPP_SCAN_MIN02,
        XPP_SCAN_NSHOTS,
        XPP_SCAN_NSTEPS,
        XPP_SCAN_SCANVAR00,
        XPP_SCAN_SCANVAR01,
        XPP_SCAN_SCANVAR02,
        XPP_TIMETOOL_AMPL,
        XPP_TIMETOOL_AMPLNXT,
        XPP_TIMETOOL_FLTPOS,
        XPP_TIMETOOL_FLTPOSFWHM,
        XPP_TIMETOOL_FLTPOS_PS,
        XPP_TIMETOOL_REFAMPL,
        XPP_USER_CCM_E,
        XPP_USER_CCM_Theta0,
        XPP_USER_FEEATT_E,
        XPP_USER_FEEATT_T,
        XPP_USER_FEEATT_T3rd,
        XPP_USER_FS3_T0_SHIFTER,
        XPP_USER_LAS_E0,
        XPP_USER_LAS_E02,
        XPP_USER_LAS_EVR0_GATE,
        XPP_USER_LAS_EVR0_OSC,
        XPP_USER_LAS_E_LEAK,
        XPP_USER_LAS_E_LEAK2,
        XPP_USER_LAS_E_PULSE,
        XPP_USER_LAS_E_PULSE2,
        XPP_USER_LAS_FS3_MAX,
        XPP_USER_LAS_FS3_MIN,
        XPP_USER_LAS_SDG0,
        XPP_USER_LAS_T0_MONITOR,
        XPP_USER_LAS_TIME_DELAY,
        XPP_USER_LOM_E,
        XPP_USER_LOM_EC,
        XPP_USER_LXT,
        XPP_USER_LXTTC,
        XPP_USER_ROB_AZ,
        XPP_USER_ROB_EL,
        XPP_USER_ROB_J1,
        XPP_USER_ROB_J2,
        XPP_USER_ROB_J3,
        XPP_USER_ROB_J4,
        XPP_USER_ROB_J5,
        XPP_USER_ROB_J6,
        XPP_USER_ROB_R,
        XPP_USER_ROB_RX,
        XPP_USER_ROB_RY,
        XPP_USER_ROB_RZ,
        XPP_USER_ROB_X,
        XPP_USER_ROB_Y,
        XPP_USER_ROB_Z,
        XPP_USER_VIT_TD,
        XPP_USR_GCC_01_PMON,
        XPP_USR_GCC_02_PMON,
        XPP_USR_GPI_01_PMON,
        XPP_USR_GPI_02_PMON,
        XPP_USR_IPM_01_ChargeAmpRangeCH0,
        XPP_USR_IPM_01_ChargeAmpRangeCH1,
        XPP_USR_IPM_01_ChargeAmpRangeCH2,
        XPP_USR_IPM_01_ChargeAmpRangeCH3,
        XPP_USR_IPM_01_DiodeBias,
        XPP_USR_LPW_01_DATA_PRI,
        XPP_USR_LPW_01_GETAOSCALE,
        XPP_USR_LPW_01_GETGAINFACTOR,
        XPP_USR_LPW_01_GETRANGE,
        XPP_USR_MMN_01_RBV,
        XPP_USR_MMN_02_RBV,
        XPP_USR_MMN_03_RBV,
        XPP_USR_MMN_04_RBV,
        XPP_USR_MMS_01_RBV,
        XPP_USR_MMS_02_RBV,
        XPP_USR_MMS_03_RBV,
        XPP_USR_MMS_04_RBV,
        XPP_USR_MMS_05_RBV,
        XPP_USR_MMS_06_RBV,
        XPP_USR_MMS_17_RBV,
        XPP_USR_OXY_01_ANALOGIN,
        XPP_USR_OXY_01_OFFSET,
        XPP_USR_OXY_01_SCALE,
        XPP_USR_TCT_01_GET_SOLL_1,
        XPP_USR_TCT_01_GET_TEMP_A,
        XPP_USR_TCT_01_GET_TEMP_B,
        XPP_USR_TCT_01_GET_TEMP_C,
        XPP_USR_TCT_01_GET_TEMP_D,
        XPP_USR_TCT_01_PUT_SOLL_1,
        XPP_USR_ao1_0,
        XPP_USR_ao1_1,
        XPP_USR_ao1_10,
        XPP_USR_ao1_11,
        XPP_USR_ao1_12,
        XPP_USR_ao1_13,
        XPP_USR_ao1_14,
        XPP_USR_ao1_15,
        XPP_USR_ao1_2,
        XPP_USR_ao1_3,
        XPP_USR_ao1_4,
        XPP_USR_ao1_5,
        XPP_USR_ao1_6,
        XPP_USR_ao1_7,
        XPP_USR_ao1_8,
        XPP_USR_ao1_9,
#else // defined(EPICSSTREAMID_UNITTEST)
        XPP_VARS_FLOAT_02,
        XPP_VARS_FLOAT_03,
        XPP_VARS_FLOAT_04,
        XPP_VARS_FLOAT_05,
        XPP_VARS_FLOAT_06,
        XPP_VARS_FLOAT_07,
        XPP_VARS_FLOAT_08,
        XPP_VARS_FLOAT_09,
        XPP_VARS_FLOAT_10,
        XPP_VARS_STRING_01,
        XPP_VARS_STRING_02,
        XPP_VARS_STRING_03,
        XPP_VARS_STRING_04,
        XPP_VARS_STRING_05,
        XPP_VARS_STRING_06,
        XPP_VARS_STRING_07,
        XPP_VARS_STRING_08,
        XPP_VARS_STRING_09,
#endif
    };

  EpicsDef()
   {
#ifndef EPICSSTREAMID_UNITTEST
        NameVec.push_back({"HX2:DVD:GCC:01:PMON",Name::DOUBLE});
        NameVec.push_back({"HX2:DVD:GPI:01:PMON",Name::DOUBLE});
        NameVec.push_back({"HX2:DVD:PIP:01:VMON",Name::DOUBLE});
        NameVec.push_back({"HX2:SB1:IPM:01:ChargeAmpRangeCH0",Name::INT64});
        NameVec.push_back({"HX2:SB1:IPM:01:DiodeBias",Name::DOUBLE});
        NameVec.push_back({"HX2:SB1:IPM:02:ChargeAmpRangeCH0",Name::INT64});
        NameVec.push_back({"HX2:SB1:IPM:02:DiodeBias",Name::DOUBLE});
        NameVec.push_back({"HX2:SB1:JAWS:ACTUAL_XCENTER",Name::DOUBLE});
        NameVec.push_back({"HX2:SB1:JAWS:ACTUAL_XWIDTH",Name::DOUBLE});
        NameVec.push_back({"HX2:SB1:JAWS:ACTUAL_YCENTER",Name::DOUBLE});
        NameVec.push_back({"HX2:SB1:JAWS:ACTUAL_YWIDTH",Name::DOUBLE});
        NameVec.push_back({"HX2:SB1:MMS:02.RBV",Name::DOUBLE});
        NameVec.push_back({"HX2:SB1:MMS:03.RBV",Name::DOUBLE});
        NameVec.push_back({"HX2:SB1:MMS:04.RBV",Name::DOUBLE});
        NameVec.push_back({"HX2:SB1:MMS:05.RBV",Name::DOUBLE});
        NameVec.push_back({"HX2:SB1:MMS:06.RBV",Name::DOUBLE});
        NameVec.push_back({"HX2:SB1:MMS:06.RRBV",Name::INT64});
        NameVec.push_back({"HX2:SB1:MMS:07.RBV",Name::DOUBLE});
        NameVec.push_back({"HX2:SB1:MMS:07.RRBV",Name::INT64});
        NameVec.push_back({"HX2:SB1:MMS:08.RBV",Name::DOUBLE});
        NameVec.push_back({"HX2:SB1:MMS:08.RRBV",Name::INT64});
        NameVec.push_back({"HX2:UVD:GCC:01:PMON",Name::DOUBLE});
        NameVec.push_back({"HX2:UVD:GPI:01:PMON",Name::DOUBLE});
        NameVec.push_back({"HX2:UVD:PIP:01:VMON",Name::DOUBLE});
        NameVec.push_back({"HX3:DVD:GPI:01:PMON",Name::DOUBLE});
        NameVec.push_back({"HX3:DVD:PIP:01:VMON",Name::DOUBLE});
        NameVec.push_back({"HX3:MON:GCC:01:PMON",Name::DOUBLE});
        NameVec.push_back({"HX3:MON:GCC:02:PMON",Name::DOUBLE});
        NameVec.push_back({"HX3:MON:GPI:01:PMON",Name::DOUBLE});
        NameVec.push_back({"HX3:MON:PIP:01:VMON",Name::DOUBLE});
        NameVec.push_back({"HX3:MON:PIP:02:VMON",Name::DOUBLE});
        NameVec.push_back({"HX3:MON:PIP:03:VMON",Name::DOUBLE});
        NameVec.push_back({"LAS:FS3:Angle:Shift:Ramp:Target",Name::DOUBLE});
        NameVec.push_back({"LAS:FS3:REG:Angle:Shift:rd",Name::DOUBLE});
        NameVec.push_back({"LAS:FS3:REG:kp_vcxo:rd",Name::INT64});
        NameVec.push_back({"LAS:FS3:VIT:FS_CTR_TIME",Name::DOUBLE});
        NameVec.push_back({"LAS:FS3:VIT:FS_TGT_TIME",Name::DOUBLE});
        NameVec.push_back({"LAS:FS3:WAVE:Signal:T_AVG",Name::DOUBLE});
        NameVec.push_back({"LAS:FS3:alldiff_fs",Name::DOUBLE});
        NameVec.push_back({"LAS:FS3:alldiff_fs:RMS",Name::DOUBLE});
        NameVec.push_back({"LAS:R54:EVR:27:CTRL.DG0C",Name::INT64});
        NameVec.push_back({"LAS:R54:EVR:27:CTRL.DG0D",Name::DOUBLE});
        NameVec.push_back({"LAS:R54:EVR:27:CTRL.DG1C",Name::INT64});
        NameVec.push_back({"LAS:R54:EVR:27:CTRL.DG1D",Name::DOUBLE});
        NameVec.push_back({"LAS:XPP:DDG:01:aDelayAO",Name::DOUBLE});
        NameVec.push_back({"PLC:XPP:LSS:C114I",Name::INT64});
        NameVec.push_back({"PLC:XPP:LSS:C116I",Name::INT64});
        NameVec.push_back({"PLC:XPP:LSS:C120I",Name::INT64});
        NameVec.push_back({"PLC:XPP:LSS:C122I",Name::INT64});
        NameVec.push_back({"PLC:XPP:LSS:C124I",Name::INT64});
        NameVec.push_back({"PLC:XPP:LSS:C126I",Name::INT64});
        NameVec.push_back({"ROOM:BSY0:1:OUTSIDETEMP",Name::DOUBLE});
        NameVec.push_back({"STEP:FEE1:441:MOTR.VAL",Name::DOUBLE});
        NameVec.push_back({"STEP:FEE1:442:MOTR.VAL",Name::DOUBLE});
        NameVec.push_back({"STEP:FEE1:443:MOTR.VAL",Name::DOUBLE});
        NameVec.push_back({"STEP:FEE1:444:MOTR.VAL",Name::DOUBLE});
        NameVec.push_back({"STEP:FEE1:445:MOTR.VAL",Name::DOUBLE});
        NameVec.push_back({"STEP:FEE1:446:MOTR.VAL",Name::DOUBLE});
        NameVec.push_back({"STEP:FEE1:447:MOTR.VAL",Name::DOUBLE});
        NameVec.push_back({"XPP:ATT:COM:R3_CUR",Name::DOUBLE});
        NameVec.push_back({"XPP:ATT:COM:R_CUR",Name::DOUBLE});
        NameVec.push_back({"XPP:ATT:COM:T_CALC.VALE",Name::DOUBLE});
        NameVec.push_back({"XPP:GON:MMS:01.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:GON:MMS:02.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:GON:MMS:03.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:GON:MMS:04.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:GON:MMS:05.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:GON:MMS:06.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:GON:MMS:07.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:GON:MMS:08.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:GON:MMS:09.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:GON:MMS:10.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:GON:MMS:11.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:GON:MMS:12.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:GON:MMS:13.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:GON:MMS:14.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:GON:MMS:15.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:GON:MMS:16.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:IPM1:TARGET_Y:STATE",Name::INT64});
        NameVec.push_back({"XPP:IPM2:TARGET_Y:STATE",Name::INT64});
        NameVec.push_back({"XPP:IPM3:TARGET_Y:STATE",Name::INT64});
        NameVec.push_back({"XPP:LAS:MMN:01.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:LAS:MMN:02.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:LAS:MMN:03.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:LAS:MMN:05.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:LAS:MMN:06.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:LAS:MMN:08.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:LAS:MMN:09.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:LAS:MMN:10.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:LAS:MMN:11.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:LAS:MMN:12.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:LAS:MMN:13.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:LAS:MMN:14.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:LAS:MMN:15.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:LAS:MMN:16.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:MON:MMS:04.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:MON:MMS:05.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:MON:MMS:06.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:MON:MMS:07.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:MON:MMS:07.VAL",Name::DOUBLE});
        NameVec.push_back({"XPP:MON:MMS:08.ACCL",Name::DOUBLE});
        NameVec.push_back({"XPP:MON:MMS:08.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:MON:MMS:08.VAL",Name::DOUBLE});
        NameVec.push_back({"XPP:MON:MMS:08:C1",Name::INT64});
        NameVec.push_back({"XPP:MON:MMS:08:C2",Name::INT64});
        NameVec.push_back({"XPP:MON:MMS:08:EL",Name::INT64});
        NameVec.push_back({"XPP:MON:MMS:09.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:MON:MMS:10.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:MON:MMS:11.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:MON:MMS:12.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:MON:MMS:13.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:MON:MMS:13.VAL",Name::DOUBLE});
        NameVec.push_back({"XPP:MON:MMS:14.ACCL",Name::DOUBLE});
        NameVec.push_back({"XPP:MON:MMS:14.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:MON:MMS:14.VAL",Name::DOUBLE});
        NameVec.push_back({"XPP:MON:MMS:14:C1",Name::INT64});
        NameVec.push_back({"XPP:MON:MMS:14:C2",Name::INT64});
        NameVec.push_back({"XPP:MON:MMS:14:EL",Name::INT64});
        NameVec.push_back({"XPP:MON:MMS:15.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:MON:MMS:16.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:MON:MMS:17.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:MON:MMS:18.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:MON:MMS:19.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:MON:MMS:20.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:MON:MMS:22.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:MON:MMS:23.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:MON:MPZ:07A:POSITIONGET",Name::DOUBLE});
        NameVec.push_back({"XPP:MON:MPZ:08:POSITIONGET",Name::DOUBLE});
        NameVec.push_back({"XPP:SB2:GCC:01:PMON",Name::DOUBLE});
        NameVec.push_back({"XPP:SB2:GPI:01:PMON",Name::DOUBLE});
        NameVec.push_back({"XPP:SB2:IPM:01:ChargeAmpRangeCH0",Name::INT64});
        NameVec.push_back({"XPP:SB2:IPM:01:DiodeBias",Name::DOUBLE});
        NameVec.push_back({"XPP:SB2:MMS:01.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:SB2:MMS:01.RRBV",Name::INT64});
        NameVec.push_back({"XPP:SB2:MMS:02.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:SB2:MMS:02.RRBV",Name::INT64});
        NameVec.push_back({"XPP:SB2:MMS:03.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:SB2:MMS:03.RRBV",Name::INT64});
        NameVec.push_back({"XPP:SB2:MMS:05.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:SB2:MMS:06.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:SB2:MMS:07.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:SB2:MMS:08.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:SB2:MMS:09.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:SB2:MMS:10.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:SB2:MMS:11.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:SB2:MMS:12.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:SB2:MMS:13.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:SB2:MMS:14.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:SB2:MMS:15.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:SB2:MMS:17.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:SB2:MMS:17.RRBV",Name::INT64});
        NameVec.push_back({"XPP:SB2:MMS:18.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:SB2:MMS:18.RRBV",Name::INT64});
        NameVec.push_back({"XPP:SB2:MMS:19.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:SB2:MMS:19.RRBV",Name::INT64});
        NameVec.push_back({"XPP:SB2:MMS:20.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:SB2:MMS:20.RRBV",Name::INT64});
        NameVec.push_back({"XPP:SB2:MMS:21.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:SB2:MMS:21.RRBV",Name::INT64});
        NameVec.push_back({"XPP:SB2:MMS:22.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:SB2:MMS:22.RRBV",Name::INT64});
        NameVec.push_back({"XPP:SB2:MMS:23.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:SB2:MMS:23.RRBV",Name::INT64});
        NameVec.push_back({"XPP:SB2:MMS:24.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:SB2:MMS:24.RRBV",Name::INT64});
        NameVec.push_back({"XPP:SB2:MMS:25.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:SB2:MMS:25.RRBV",Name::INT64});
        NameVec.push_back({"XPP:SB2:MMS:26.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:SB2:MMS:26.RRBV",Name::INT64});
        NameVec.push_back({"XPP:SB2:MMS:27.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:SB2:MMS:27.RRBV",Name::INT64});
        NameVec.push_back({"XPP:SB2H:JAWS:ACTUAL_XCENTER",Name::DOUBLE});
        NameVec.push_back({"XPP:SB2H:JAWS:ACTUAL_XWIDTH",Name::DOUBLE});
        NameVec.push_back({"XPP:SB2H:JAWS:ACTUAL_YCENTER",Name::DOUBLE});
        NameVec.push_back({"XPP:SB2H:JAWS:ACTUAL_YWIDTH",Name::DOUBLE});
        NameVec.push_back({"XPP:SB2L:JAWS:ACTUAL_XCENTER",Name::DOUBLE});
        NameVec.push_back({"XPP:SB2L:JAWS:ACTUAL_XWIDTH",Name::DOUBLE});
        NameVec.push_back({"XPP:SB2L:JAWS:ACTUAL_YCENTER",Name::DOUBLE});
        NameVec.push_back({"XPP:SB2L:JAWS:ACTUAL_YWIDTH",Name::DOUBLE});
        NameVec.push_back({"XPP:SB3:CLF:01.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:SB3:CLZ:01.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:SB3:GCC:01:PMON",Name::DOUBLE});
        NameVec.push_back({"XPP:SB3:GCC:02:PMON",Name::DOUBLE});
        NameVec.push_back({"XPP:SB3:GPI:01:PMON",Name::DOUBLE});
        NameVec.push_back({"XPP:SB3:GPI:02:PMON",Name::DOUBLE});
        NameVec.push_back({"XPP:SB3:IPM:01:ChargeAmpRangeCH0",Name::INT64});
        NameVec.push_back({"XPP:SB3:IPM:01:DiodeBias",Name::DOUBLE});
        NameVec.push_back({"XPP:SB3:JAWS:ACTUAL_XCENTER",Name::DOUBLE});
        NameVec.push_back({"XPP:SB3:JAWS:ACTUAL_XWIDTH",Name::DOUBLE});
        NameVec.push_back({"XPP:SB3:JAWS:ACTUAL_YCENTER",Name::DOUBLE});
        NameVec.push_back({"XPP:SB3:JAWS:ACTUAL_YWIDTH",Name::DOUBLE});
        NameVec.push_back({"XPP:SB3:MMS:01.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:SB3:MMS:02.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:SB3:MMS:03.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:SB3:MMS:04.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:SB3:MMS:05.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:SB3:MMS:06.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:SB3:MMS:07.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:SB3:MMS:08.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:SB3:MMS:09.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:SB3:MMS:10.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:SB3:MMS:11.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:SB3:MMS:11.RRBV",Name::INT64});
        NameVec.push_back({"XPP:SB3:MMS:12.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:SB3:MMS:12.RRBV",Name::INT64});
        NameVec.push_back({"XPP:SB3:MMS:13.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:SB3:MMS:13.RRBV",Name::INT64});
        NameVec.push_back({"XPP:SB3:MMS:14.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:SB3:MMS:14.RRBV",Name::INT64});
        NameVec.push_back({"XPP:SB3:MMS:15.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:SB3:MMS:15.RRBV",Name::INT64});
        NameVec.push_back({"XPP:SB3:PIP:01:VMON",Name::DOUBLE});
        NameVec.push_back({"XPP:SB4:GCC:01:PMON",Name::DOUBLE});
        NameVec.push_back({"XPP:SB4:GPI:01:PMON",Name::DOUBLE});
        NameVec.push_back({"XPP:SB4:IPM:01:ChargeAmpRangeCH0",Name::INT64});
        NameVec.push_back({"XPP:SB4:IPM:01:ChargeAmpRangeCH1",Name::INT64});
        NameVec.push_back({"XPP:SB4:IPM:01:ChargeAmpRangeCH2",Name::INT64});
        NameVec.push_back({"XPP:SB4:IPM:01:ChargeAmpRangeCH3",Name::INT64});
        NameVec.push_back({"XPP:SB4:IPM:01:DiodeBias",Name::DOUBLE});
        NameVec.push_back({"XPP:SB4:USR:MMS:42",Name::DOUBLE});
        NameVec.push_back({"XPP:SB4:USR:MMS:43",Name::DOUBLE});
        NameVec.push_back({"XPP:SCAN:ISSCAN",Name::INT64});
        NameVec.push_back({"XPP:SCAN:ISTEP",Name::INT64});
        NameVec.push_back({"XPP:SCAN:MAX00",Name::DOUBLE});
        NameVec.push_back({"XPP:SCAN:MAX01",Name::DOUBLE});
        NameVec.push_back({"XPP:SCAN:MAX02",Name::DOUBLE});
        NameVec.push_back({"XPP:SCAN:MIN00",Name::DOUBLE});
        NameVec.push_back({"XPP:SCAN:MIN01",Name::DOUBLE});
        NameVec.push_back({"XPP:SCAN:MIN02",Name::DOUBLE});
        NameVec.push_back({"XPP:SCAN:NSHOTS",Name::INT64});
        NameVec.push_back({"XPP:SCAN:NSTEPS",Name::INT64});
        NameVec.push_back({"XPP:SCAN:SCANVAR00",Name::CHARSTR,1});
        NameVec.push_back({"XPP:SCAN:SCANVAR01",Name::CHARSTR,1});
        NameVec.push_back({"XPP:SCAN:SCANVAR02",Name::CHARSTR,1});
        NameVec.push_back({"XPP:TIMETOOL:AMPL",Name::DOUBLE});
        NameVec.push_back({"XPP:TIMETOOL:AMPLNXT",Name::DOUBLE});
        NameVec.push_back({"XPP:TIMETOOL:FLTPOS",Name::DOUBLE});
        NameVec.push_back({"XPP:TIMETOOL:FLTPOSFWHM",Name::DOUBLE});
        NameVec.push_back({"XPP:TIMETOOL:FLTPOS_PS",Name::DOUBLE});
        NameVec.push_back({"XPP:TIMETOOL:REFAMPL",Name::DOUBLE});
        NameVec.push_back({"XPP:USER:CCM:E",Name::DOUBLE});
        NameVec.push_back({"XPP:USER:CCM:Theta0",Name::DOUBLE});
        NameVec.push_back({"XPP:USER:FEEATT:E",Name::DOUBLE});
        NameVec.push_back({"XPP:USER:FEEATT:T",Name::DOUBLE});
        NameVec.push_back({"XPP:USER:FEEATT:T3rd",Name::DOUBLE});
        NameVec.push_back({"XPP:USER:FS3:T0_SHIFTER",Name::DOUBLE});
        NameVec.push_back({"XPP:USER:LAS:E0",Name::DOUBLE});
        NameVec.push_back({"XPP:USER:LAS:E02",Name::DOUBLE});
        NameVec.push_back({"XPP:USER:LAS:EVR0_GATE",Name::DOUBLE});
        NameVec.push_back({"XPP:USER:LAS:EVR0_OSC",Name::DOUBLE});
        NameVec.push_back({"XPP:USER:LAS:E_LEAK",Name::DOUBLE});
        NameVec.push_back({"XPP:USER:LAS:E_LEAK2",Name::DOUBLE});
        NameVec.push_back({"XPP:USER:LAS:E_PULSE",Name::DOUBLE});
        NameVec.push_back({"XPP:USER:LAS:E_PULSE2",Name::DOUBLE});
        NameVec.push_back({"XPP:USER:LAS:FS3_MAX",Name::DOUBLE});
        NameVec.push_back({"XPP:USER:LAS:FS3_MIN",Name::DOUBLE});
        NameVec.push_back({"XPP:USER:LAS:SDG0",Name::DOUBLE});
        NameVec.push_back({"XPP:USER:LAS:T0_MONITOR",Name::DOUBLE});
        NameVec.push_back({"XPP:USER:LAS:TIME_DELAY",Name::DOUBLE});
        NameVec.push_back({"XPP:USER:LOM:E",Name::DOUBLE});
        NameVec.push_back({"XPP:USER:LOM:EC",Name::DOUBLE});
        NameVec.push_back({"XPP:USER:LXT",Name::DOUBLE});
        NameVec.push_back({"XPP:USER:LXTTC",Name::DOUBLE});
        NameVec.push_back({"XPP:USER:ROB:AZ",Name::DOUBLE});
        NameVec.push_back({"XPP:USER:ROB:EL",Name::DOUBLE});
        NameVec.push_back({"XPP:USER:ROB:J1",Name::DOUBLE});
        NameVec.push_back({"XPP:USER:ROB:J2",Name::DOUBLE});
        NameVec.push_back({"XPP:USER:ROB:J3",Name::DOUBLE});
        NameVec.push_back({"XPP:USER:ROB:J4",Name::DOUBLE});
        NameVec.push_back({"XPP:USER:ROB:J5",Name::DOUBLE});
        NameVec.push_back({"XPP:USER:ROB:J6",Name::DOUBLE});
        NameVec.push_back({"XPP:USER:ROB:R",Name::DOUBLE});
        NameVec.push_back({"XPP:USER:ROB:RX",Name::DOUBLE});
        NameVec.push_back({"XPP:USER:ROB:RY",Name::DOUBLE});
        NameVec.push_back({"XPP:USER:ROB:RZ",Name::DOUBLE});
        NameVec.push_back({"XPP:USER:ROB:X",Name::DOUBLE});
        NameVec.push_back({"XPP:USER:ROB:Y",Name::DOUBLE});
        NameVec.push_back({"XPP:USER:ROB:Z",Name::DOUBLE});
        NameVec.push_back({"XPP:USER:VIT:TD",Name::DOUBLE});
        NameVec.push_back({"XPP:USR:GCC:01:PMON",Name::DOUBLE});
        NameVec.push_back({"XPP:USR:GCC:02:PMON",Name::DOUBLE});
        NameVec.push_back({"XPP:USR:GPI:01:PMON",Name::DOUBLE});
        NameVec.push_back({"XPP:USR:GPI:02:PMON",Name::DOUBLE});
        NameVec.push_back({"XPP:USR:IPM:01:ChargeAmpRangeCH0",Name::INT64});
        NameVec.push_back({"XPP:USR:IPM:01:ChargeAmpRangeCH1",Name::INT64});
        NameVec.push_back({"XPP:USR:IPM:01:ChargeAmpRangeCH2",Name::INT64});
        NameVec.push_back({"XPP:USR:IPM:01:ChargeAmpRangeCH3",Name::INT64});
        NameVec.push_back({"XPP:USR:IPM:01:DiodeBias",Name::DOUBLE});
        NameVec.push_back({"XPP:USR:LPW:01:DATA_PRI",Name::DOUBLE});
        NameVec.push_back({"XPP:USR:LPW:01:GETAOSCALE",Name::INT64});
        NameVec.push_back({"XPP:USR:LPW:01:GETGAINFACTOR",Name::DOUBLE});
        NameVec.push_back({"XPP:USR:LPW:01:GETRANGE",Name::DOUBLE});
        NameVec.push_back({"XPP:USR:MMN:01.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:USR:MMN:02.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:USR:MMN:03.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:USR:MMN:04.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:USR:MMS:01.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:USR:MMS:02.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:USR:MMS:03.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:USR:MMS:04.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:USR:MMS:05.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:USR:MMS:06.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:USR:MMS:17.RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:USR:OXY:01:ANALOGIN",Name::DOUBLE});
        NameVec.push_back({"XPP:USR:OXY:01:OFFSET",Name::DOUBLE});
        NameVec.push_back({"XPP:USR:OXY:01:SCALE",Name::DOUBLE});
        NameVec.push_back({"XPP:USR:TCT:01:GET_SOLL_1",Name::DOUBLE});
        NameVec.push_back({"XPP:USR:TCT:01:GET_TEMP_A",Name::DOUBLE});
        NameVec.push_back({"XPP:USR:TCT:01:GET_TEMP_B",Name::DOUBLE});
        NameVec.push_back({"XPP:USR:TCT:01:GET_TEMP_C",Name::DOUBLE});
        NameVec.push_back({"XPP:USR:TCT:01:GET_TEMP_D",Name::DOUBLE});
        NameVec.push_back({"XPP:USR:TCT:01:PUT_SOLL_1",Name::DOUBLE});
        NameVec.push_back({"XPP:USR:ao1:0",Name::DOUBLE});
        NameVec.push_back({"XPP:USR:ao1:1",Name::DOUBLE});
        NameVec.push_back({"XPP:USR:ao1:10",Name::DOUBLE});
        NameVec.push_back({"XPP:USR:ao1:11",Name::DOUBLE});
        NameVec.push_back({"XPP:USR:ao1:12",Name::DOUBLE});
        NameVec.push_back({"XPP:USR:ao1:13",Name::DOUBLE});
        NameVec.push_back({"XPP:USR:ao1:14",Name::DOUBLE});
        NameVec.push_back({"XPP:USR:ao1:15",Name::DOUBLE});
        NameVec.push_back({"XPP:USR:ao1:2",Name::DOUBLE});
        NameVec.push_back({"XPP:USR:ao1:3",Name::DOUBLE});
        NameVec.push_back({"XPP:USR:ao1:4",Name::DOUBLE});
        NameVec.push_back({"XPP:USR:ao1:5",Name::DOUBLE});
        NameVec.push_back({"XPP:USR:ao1:6",Name::DOUBLE});
        NameVec.push_back({"XPP:USR:ao1:7",Name::DOUBLE});
        NameVec.push_back({"XPP:USR:ao1:8",Name::DOUBLE});
        NameVec.push_back({"XPP:USR:ao1:9",Name::DOUBLE});
#else // defined(EPICSSTREAMID_UNITTEST)
        NameVec.push_back({"XPP:VARS:FLOAT:02",Name::DOUBLE});
        NameVec.push_back({"XPP:VARS:FLOAT:03",Name::DOUBLE});
        NameVec.push_back({"XPP:VARS:FLOAT:04",Name::DOUBLE});
        NameVec.push_back({"XPP:VARS:FLOAT:05",Name::DOUBLE});
        NameVec.push_back({"XPP:VARS:FLOAT:06",Name::DOUBLE});
        NameVec.push_back({"XPP:VARS:FLOAT:07",Name::DOUBLE});
        NameVec.push_back({"XPP:VARS:FLOAT:08",Name::DOUBLE});
        NameVec.push_back({"XPP:VARS:FLOAT:09",Name::DOUBLE});
        NameVec.push_back({"XPP:VARS:FLOAT:10",Name::DOUBLE});
        NameVec.push_back({"XPP:VARS:STRING:01",Name::CHARSTR,1});
        NameVec.push_back({"XPP:VARS:STRING:02",Name::CHARSTR,1});
        NameVec.push_back({"XPP:VARS:STRING:03",Name::CHARSTR,1});
        NameVec.push_back({"XPP:VARS:STRING:04",Name::CHARSTR,1});
        NameVec.push_back({"XPP:VARS:STRING:05",Name::CHARSTR,1});
        NameVec.push_back({"XPP:VARS:STRING:06",Name::CHARSTR,1});
        NameVec.push_back({"XPP:VARS:STRING:07",Name::CHARSTR,1});
        NameVec.push_back({"XPP:VARS:STRING:08",Name::CHARSTR,1});
        NameVec.push_back({"XPP:VARS:STRING:09",Name::CHARSTR,1});
#endif
   }
} EpicsDef;


class DebugIter : public XtcIterator
{
public:
    enum { Stop, Continue };
    DebugIter(Xtc* xtc, NamesLookup& namesLookup) : XtcIterator(xtc), _namesLookup(namesLookup)
    {
    }

    void get_value(int i, Name& name, DescData& descdata){
        int data_rank = name.rank();
        int data_type = name.type();

        switch(name.type()){
        case(0):{
            if(data_rank > 0){
                Array<uint8_t> arrT = descdata.get_array<uint8_t>(i);
                // printf("%s: %d, %d, %d\n",name.name(),arrT(0),arrT(1), arrT(2));
                    }
            else{
                // printf("%s: %d\n",name.name(),descdata.get_value<uint8_t>(i));
            }
            break;
        }

        case(1):{
            if(data_rank > 0){
                Array<uint16_t> arrT = descdata.get_array<uint16_t>(i);
                // printf("%s: %d, %d, %d\n",name.name(),arrT(0),arrT(1), arrT(2));
                    }
            else{
                // printf("%s: %d\n",name.name(),descdata.get_value<uint16_t>(i));
            }
            break;
        }

        case(2):{
            if(data_rank > 0){
                Array<uint32_t> arrT = descdata.get_array<uint32_t>(i);
                // printf("%s: %d, %d, %d\n",name.name(),arrT(0),arrT(1), arrT(2));
                    }
            else{
                // printf("%s: %d\n",name.name(),descdata.get_value<uint32_t>(i));
            }
            break;
        }

        case(3):{
            if(data_rank > 0){
                Array<uint64_t> arrT = descdata.get_array<uint64_t>(i);
                // printf("%s: %d, %d, %d\n",name.name(),arrT(0),arrT(1), arrT(2));
                    }
            else{
                // printf("%s: %d\n",name.name(),descdata.get_value<uint64_t>(i));
            }
            break;
        }

        case(4):{
            if(data_rank > 0){
                Array<int8_t> arrT = descdata.get_array<int8_t>(i);
                // printf("%s: %d, %d, %d\n",name.name(),arrT(0),arrT(1), arrT(2));
                    }
            else{
                // printf("%s: %d\n",name.name(),descdata.get_value<int8_t>(i));
            }
            break;
        }

        case(5):{
            if(data_rank > 0){
                Array<int16_t> arrT = descdata.get_array<int16_t>(i);
                // printf("%s: %d, %d, %d\n",name.name(),arrT(0),arrT(1), arrT(2));
                    }
            else{
                // printf("%s: %d\n",name.name(),descdata.get_value<int16_t>(i));
            }
            break;
        }

        case(6):{
            if(data_rank > 0){
                Array<int32_t> arrT = descdata.get_array<int32_t>(i);
                // printf("%s: %d, %d, %d\n",name.name(),arrT(0),arrT(1), arrT(2));
                    }
            else{
                // printf("%s: %d\n",name.name(),descdata.get_value<int32_t>(i));
            }
            break;
        }

        case(7):{
            if(data_rank > 0){
                Array<int64_t> arrT = descdata.get_array<int64_t>(i);
                // printf("%s: %d, %d, %d\n",name.name(),arrT(0),arrT(1), arrT(2));
                    }
            else{
                // printf("%s: %d\n",name.name(),descdata.get_value<int64_t>(i));
            }
            break;
        }

        case(8):{
            if(data_rank > 0){
                Array<float> arrT = descdata.get_array<float>(i);
                // printf("%s: %f, %f\n",name.name(),arrT(0),arrT(1));
                    }
            else{
                // printf("%s: %f\n",name.name(),descdata.get_value<float>(i));
            }
            break;
        }

        case(9):{
            if(data_rank > 0){
                Array<double> arrT = descdata.get_array<double>(i);
                // printf("%s: %f, %f, %f\n",name.name(),arrT(0),arrT(1), arrT(2));
                    }
            else{
                // printf("%s: %f\n",name.name(),descdata.get_value<double>(i));
            }
            break;
        }

        }

    }

    int process(Xtc* xtc)
    {
        // printf("found typeid %s\n",XtcData::TypeId::name(xtc->contains.id()));
        switch (xtc->contains.id()) {

        case (TypeId::Names): {
            // printf("Names pointer is %p\n", xtc);
            break;
        }
        case (TypeId::Parent): {
            iterate(xtc);
            break;
        }
        case (TypeId::ShapesData): {
            ShapesData& shapesdata = *(ShapesData*)xtc;
            // lookup the index of the names we are supposed to use
            NamesId& namesId = shapesdata.namesId();
            DescData descdata(shapesdata, _namesLookup[namesId]);
            Names& names = descdata.nameindex().names();
	    //   printf("Found %d names\n",names.num());
            // printf("data shapes extents %d %d %d\n", shapesdata.data().extent,
                   // shapesdata.shapes().extent, sizeof(double));
            for (unsigned i = 0; i < names.num(); i++) {
                Name& name = names.get(i);
                get_value(i, name, descdata);
            }
            break;
        }
        default:
            break;
        }
        return Continue;
    }
    NamesLookup& _namesLookup;
    void printOffset(const char* str, void* base, void* ptr) {
        printf("***%s at offset %li addr %p\n",str,(char*)ptr-(char*)base,ptr);
    }
};

void epicsExample(Xtc& parent, NamesLookup& namesLookup, NamesId& namesId)
{ 
    CreateData epics(parent, namesLookup, namesId);

#ifndef EPICSSTREAMID_UNITTEST
    epics.set_value(EpicsDef::HX2_DVD_GCC_01_PMON, (double)41.0);
    epics.set_value(EpicsDef::HX2_DVD_GPI_01_PMON, (double)41.0);
    epics.set_value(EpicsDef::HX2_DVD_PIP_01_VMON, (double)41.0);
    epics.set_value(EpicsDef::HX2_SB1_IPM_01_ChargeAmpRangeCH0, (int64_t) 42);
    epics.set_value(EpicsDef::HX2_SB1_IPM_01_DiodeBias, (double)41.0);
    epics.set_value(EpicsDef::HX2_SB1_IPM_02_ChargeAmpRangeCH0, (int64_t) 42);
    epics.set_value(EpicsDef::HX2_SB1_IPM_02_DiodeBias, (double)41.0);
    epics.set_value(EpicsDef::HX2_SB1_JAWS_ACTUAL_XCENTER, (double)41.0);
    epics.set_value(EpicsDef::HX2_SB1_JAWS_ACTUAL_XWIDTH, (double)41.0);
    epics.set_value(EpicsDef::HX2_SB1_JAWS_ACTUAL_YCENTER, (double)41.0);
    epics.set_value(EpicsDef::HX2_SB1_JAWS_ACTUAL_YWIDTH, (double)41.0);
    epics.set_value(EpicsDef::HX2_SB1_MMS_02_RBV, (double)41.0);
    epics.set_value(EpicsDef::HX2_SB1_MMS_03_RBV, (double)41.0);
    epics.set_value(EpicsDef::HX2_SB1_MMS_04_RBV, (double)41.0);
    epics.set_value(EpicsDef::HX2_SB1_MMS_05_RBV, (double)41.0);
    epics.set_value(EpicsDef::HX2_SB1_MMS_06_RBV, (double)41.0);
    epics.set_value(EpicsDef::HX2_SB1_MMS_06_RRBV, (int64_t) 42);
    epics.set_value(EpicsDef::HX2_SB1_MMS_07_RBV, (double)41.0);
    epics.set_value(EpicsDef::HX2_SB1_MMS_07_RRBV, (int64_t) 42);
    epics.set_value(EpicsDef::HX2_SB1_MMS_08_RBV, (double)41.0);
    epics.set_value(EpicsDef::HX2_SB1_MMS_08_RRBV, (int64_t) 42);
    epics.set_value(EpicsDef::HX2_UVD_GCC_01_PMON, (double)41.0);
    epics.set_value(EpicsDef::HX2_UVD_GPI_01_PMON, (double)41.0);
    epics.set_value(EpicsDef::HX2_UVD_PIP_01_VMON, (double)41.0);
    epics.set_value(EpicsDef::HX3_DVD_GPI_01_PMON, (double)41.0);
    epics.set_value(EpicsDef::HX3_DVD_PIP_01_VMON, (double)41.0);
    epics.set_value(EpicsDef::HX3_MON_GCC_01_PMON, (double)41.0);
    epics.set_value(EpicsDef::HX3_MON_GCC_02_PMON, (double)41.0);
    epics.set_value(EpicsDef::HX3_MON_GPI_01_PMON, (double)41.0);
    epics.set_value(EpicsDef::HX3_MON_PIP_01_VMON, (double)41.0);
    epics.set_value(EpicsDef::HX3_MON_PIP_02_VMON, (double)41.0);
    epics.set_value(EpicsDef::HX3_MON_PIP_03_VMON, (double)41.0);
    epics.set_value(EpicsDef::LAS_FS3_Angle_Shift_Ramp_Target, (double)41.0);
    epics.set_value(EpicsDef::LAS_FS3_REG_Angle_Shift_rd, (double)41.0);
    epics.set_value(EpicsDef::LAS_FS3_REG_kp_vcxo_rd, (int64_t) 42);
    epics.set_value(EpicsDef::LAS_FS3_VIT_FS_CTR_TIME, (double)41.0);
    epics.set_value(EpicsDef::LAS_FS3_VIT_FS_TGT_TIME, (double)41.0);
    epics.set_value(EpicsDef::LAS_FS3_WAVE_Signal_T_AVG, (double)41.0);
    epics.set_value(EpicsDef::LAS_FS3_alldiff_fs, (double)41.0);
    epics.set_value(EpicsDef::LAS_FS3_alldiff_fs_RMS, (double)41.0);
    epics.set_value(EpicsDef::LAS_R54_EVR_27_CTRL_DG0C, (int64_t) 42);
    epics.set_value(EpicsDef::LAS_R54_EVR_27_CTRL_DG0D, (double)41.0);
    epics.set_value(EpicsDef::LAS_R54_EVR_27_CTRL_DG1C, (int64_t) 42);
    epics.set_value(EpicsDef::LAS_R54_EVR_27_CTRL_DG1D, (double)41.0);
    epics.set_value(EpicsDef::LAS_XPP_DDG_01_aDelayAO, (double)41.0);
    epics.set_value(EpicsDef::PLC_XPP_LSS_C114I, (int64_t) 42);
    epics.set_value(EpicsDef::PLC_XPP_LSS_C116I, (int64_t) 42);
    epics.set_value(EpicsDef::PLC_XPP_LSS_C120I, (int64_t) 42);
    epics.set_value(EpicsDef::PLC_XPP_LSS_C122I, (int64_t) 42);
    epics.set_value(EpicsDef::PLC_XPP_LSS_C124I, (int64_t) 42);
    epics.set_value(EpicsDef::PLC_XPP_LSS_C126I, (int64_t) 42);
    epics.set_value(EpicsDef::ROOM_BSY0_1_OUTSIDETEMP, (double)41.0);
    epics.set_value(EpicsDef::STEP_FEE1_441_MOTR_VAL, (double)41.0);
    epics.set_value(EpicsDef::STEP_FEE1_442_MOTR_VAL, (double)41.0);
    epics.set_value(EpicsDef::STEP_FEE1_443_MOTR_VAL, (double)41.0);
    epics.set_value(EpicsDef::STEP_FEE1_444_MOTR_VAL, (double)41.0);
    epics.set_value(EpicsDef::STEP_FEE1_445_MOTR_VAL, (double)41.0);
    epics.set_value(EpicsDef::STEP_FEE1_446_MOTR_VAL, (double)41.0);
    epics.set_value(EpicsDef::STEP_FEE1_447_MOTR_VAL, (double)41.0);
    epics.set_value(EpicsDef::XPP_ATT_COM_R3_CUR, (double)41.0);
    epics.set_value(EpicsDef::XPP_ATT_COM_R_CUR, (double)41.0);
    epics.set_value(EpicsDef::XPP_ATT_COM_T_CALC_VALE, (double)41.0);
    epics.set_value(EpicsDef::XPP_GON_MMS_01_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_GON_MMS_02_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_GON_MMS_03_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_GON_MMS_04_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_GON_MMS_05_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_GON_MMS_06_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_GON_MMS_07_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_GON_MMS_08_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_GON_MMS_09_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_GON_MMS_10_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_GON_MMS_11_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_GON_MMS_12_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_GON_MMS_13_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_GON_MMS_14_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_GON_MMS_15_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_GON_MMS_16_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_IPM1_TARGET_Y_STATE, (int64_t) 42);
    epics.set_value(EpicsDef::XPP_IPM2_TARGET_Y_STATE, (int64_t) 42);
    epics.set_value(EpicsDef::XPP_IPM3_TARGET_Y_STATE, (int64_t) 42);
    epics.set_value(EpicsDef::XPP_LAS_MMN_01_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_LAS_MMN_02_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_LAS_MMN_03_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_LAS_MMN_05_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_LAS_MMN_06_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_LAS_MMN_08_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_LAS_MMN_09_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_LAS_MMN_10_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_LAS_MMN_11_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_LAS_MMN_12_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_LAS_MMN_13_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_LAS_MMN_14_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_LAS_MMN_15_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_LAS_MMN_16_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_MON_MMS_04_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_MON_MMS_05_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_MON_MMS_06_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_MON_MMS_07_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_MON_MMS_07_VAL, (double)41.0);
    epics.set_value(EpicsDef::XPP_MON_MMS_08_ACCL, (double)41.0);
    epics.set_value(EpicsDef::XPP_MON_MMS_08_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_MON_MMS_08_VAL, (double)41.0);
    epics.set_value(EpicsDef::XPP_MON_MMS_08_C1, (int64_t) 42);
    epics.set_value(EpicsDef::XPP_MON_MMS_08_C2, (int64_t) 42);
    epics.set_value(EpicsDef::XPP_MON_MMS_08_EL, (int64_t) 42);
    epics.set_value(EpicsDef::XPP_MON_MMS_09_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_MON_MMS_10_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_MON_MMS_11_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_MON_MMS_12_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_MON_MMS_13_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_MON_MMS_13_VAL, (double)41.0);
    epics.set_value(EpicsDef::XPP_MON_MMS_14_ACCL, (double)41.0);
    epics.set_value(EpicsDef::XPP_MON_MMS_14_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_MON_MMS_14_VAL, (double)41.0);
    epics.set_value(EpicsDef::XPP_MON_MMS_14_C1, (int64_t) 42);
    epics.set_value(EpicsDef::XPP_MON_MMS_14_C2, (int64_t) 42);
    epics.set_value(EpicsDef::XPP_MON_MMS_14_EL, (int64_t) 42);
    epics.set_value(EpicsDef::XPP_MON_MMS_15_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_MON_MMS_16_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_MON_MMS_17_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_MON_MMS_18_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_MON_MMS_19_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_MON_MMS_20_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_MON_MMS_22_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_MON_MMS_23_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_MON_MPZ_07A_POSITIONGET, (double)41.0);
    epics.set_value(EpicsDef::XPP_MON_MPZ_08_POSITIONGET, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB2_GCC_01_PMON, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB2_GPI_01_PMON, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB2_IPM_01_ChargeAmpRangeCH0, (int64_t) 42);
    epics.set_value(EpicsDef::XPP_SB2_IPM_01_DiodeBias, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB2_MMS_01_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB2_MMS_01_RRBV, (int64_t) 42);
    epics.set_value(EpicsDef::XPP_SB2_MMS_02_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB2_MMS_02_RRBV, (int64_t) 42);
    epics.set_value(EpicsDef::XPP_SB2_MMS_03_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB2_MMS_03_RRBV, (int64_t) 42);
    epics.set_value(EpicsDef::XPP_SB2_MMS_05_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB2_MMS_06_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB2_MMS_07_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB2_MMS_08_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB2_MMS_09_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB2_MMS_10_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB2_MMS_11_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB2_MMS_12_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB2_MMS_13_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB2_MMS_14_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB2_MMS_15_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB2_MMS_17_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB2_MMS_17_RRBV, (int64_t) 42);
    epics.set_value(EpicsDef::XPP_SB2_MMS_18_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB2_MMS_18_RRBV, (int64_t) 42);
    epics.set_value(EpicsDef::XPP_SB2_MMS_19_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB2_MMS_19_RRBV, (int64_t) 42);
    epics.set_value(EpicsDef::XPP_SB2_MMS_20_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB2_MMS_20_RRBV, (int64_t) 42);
    epics.set_value(EpicsDef::XPP_SB2_MMS_21_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB2_MMS_21_RRBV, (int64_t) 42);
    epics.set_value(EpicsDef::XPP_SB2_MMS_22_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB2_MMS_22_RRBV, (int64_t) 42);
    epics.set_value(EpicsDef::XPP_SB2_MMS_23_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB2_MMS_23_RRBV, (int64_t) 42);
    epics.set_value(EpicsDef::XPP_SB2_MMS_24_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB2_MMS_24_RRBV, (int64_t) 42);
    epics.set_value(EpicsDef::XPP_SB2_MMS_25_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB2_MMS_25_RRBV, (int64_t) 42);
    epics.set_value(EpicsDef::XPP_SB2_MMS_26_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB2_MMS_26_RRBV, (int64_t) 42);
    epics.set_value(EpicsDef::XPP_SB2_MMS_27_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB2_MMS_27_RRBV, (int64_t) 42);
    epics.set_value(EpicsDef::XPP_SB2H_JAWS_ACTUAL_XCENTER, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB2H_JAWS_ACTUAL_XWIDTH, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB2H_JAWS_ACTUAL_YCENTER, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB2H_JAWS_ACTUAL_YWIDTH, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB2L_JAWS_ACTUAL_XCENTER, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB2L_JAWS_ACTUAL_XWIDTH, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB2L_JAWS_ACTUAL_YCENTER, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB2L_JAWS_ACTUAL_YWIDTH, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB3_CLF_01_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB3_CLZ_01_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB3_GCC_01_PMON, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB3_GCC_02_PMON, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB3_GPI_01_PMON, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB3_GPI_02_PMON, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB3_IPM_01_ChargeAmpRangeCH0, (int64_t) 42);
    epics.set_value(EpicsDef::XPP_SB3_IPM_01_DiodeBias, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB3_JAWS_ACTUAL_XCENTER, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB3_JAWS_ACTUAL_XWIDTH, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB3_JAWS_ACTUAL_YCENTER, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB3_JAWS_ACTUAL_YWIDTH, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB3_MMS_01_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB3_MMS_02_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB3_MMS_03_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB3_MMS_04_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB3_MMS_05_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB3_MMS_06_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB3_MMS_07_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB3_MMS_08_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB3_MMS_09_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB3_MMS_10_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB3_MMS_11_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB3_MMS_11_RRBV, (int64_t) 42);
    epics.set_value(EpicsDef::XPP_SB3_MMS_12_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB3_MMS_12_RRBV, (int64_t) 42);
    epics.set_value(EpicsDef::XPP_SB3_MMS_13_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB3_MMS_13_RRBV, (int64_t) 42);
    epics.set_value(EpicsDef::XPP_SB3_MMS_14_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB3_MMS_14_RRBV, (int64_t) 42);
    epics.set_value(EpicsDef::XPP_SB3_MMS_15_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB3_MMS_15_RRBV, (int64_t) 42);
    epics.set_value(EpicsDef::XPP_SB3_PIP_01_VMON, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB4_GCC_01_PMON, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB4_GPI_01_PMON, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB4_IPM_01_ChargeAmpRangeCH0, (int64_t) 42);
    epics.set_value(EpicsDef::XPP_SB4_IPM_01_ChargeAmpRangeCH1, (int64_t) 42);
    epics.set_value(EpicsDef::XPP_SB4_IPM_01_ChargeAmpRangeCH2, (int64_t) 42);
    epics.set_value(EpicsDef::XPP_SB4_IPM_01_ChargeAmpRangeCH3, (int64_t) 42);
    epics.set_value(EpicsDef::XPP_SB4_IPM_01_DiodeBias, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB4_USR_MMS_42, (double)41.0);
    epics.set_value(EpicsDef::XPP_SB4_USR_MMS_43, (double)41.0);
    epics.set_value(EpicsDef::XPP_SCAN_ISSCAN, (int64_t) 42);
    epics.set_value(EpicsDef::XPP_SCAN_ISTEP, (int64_t) 42);
    epics.set_value(EpicsDef::XPP_SCAN_MAX00, (double)41.0);
    epics.set_value(EpicsDef::XPP_SCAN_MAX01, (double)41.0);
    epics.set_value(EpicsDef::XPP_SCAN_MAX02, (double)41.0);
    epics.set_value(EpicsDef::XPP_SCAN_MIN00, (double)41.0);
    epics.set_value(EpicsDef::XPP_SCAN_MIN01, (double)41.0);
    epics.set_value(EpicsDef::XPP_SCAN_MIN02, (double)41.0);
    epics.set_value(EpicsDef::XPP_SCAN_NSHOTS, (int64_t) 42);
    epics.set_value(EpicsDef::XPP_SCAN_NSTEPS, (int64_t) 42);
    epics.set_string(EpicsDef::XPP_SCAN_SCANVAR00, "Test String");
    epics.set_string(EpicsDef::XPP_SCAN_SCANVAR01, "Test String");
    epics.set_string(EpicsDef::XPP_SCAN_SCANVAR02, "Test String");
    epics.set_value(EpicsDef::XPP_TIMETOOL_AMPL, (double)41.0);
    epics.set_value(EpicsDef::XPP_TIMETOOL_AMPLNXT, (double)41.0);
    epics.set_value(EpicsDef::XPP_TIMETOOL_FLTPOS, (double)41.0);
    epics.set_value(EpicsDef::XPP_TIMETOOL_FLTPOSFWHM, (double)41.0);
    epics.set_value(EpicsDef::XPP_TIMETOOL_FLTPOS_PS, (double)41.0);
    epics.set_value(EpicsDef::XPP_TIMETOOL_REFAMPL, (double)41.0);
    epics.set_value(EpicsDef::XPP_USER_CCM_E, (double)41.0);
    epics.set_value(EpicsDef::XPP_USER_CCM_Theta0, (double)41.0);
    epics.set_value(EpicsDef::XPP_USER_FEEATT_E, (double)41.0);
    epics.set_value(EpicsDef::XPP_USER_FEEATT_T, (double)41.0);
    epics.set_value(EpicsDef::XPP_USER_FEEATT_T3rd, (double)41.0);
    epics.set_value(EpicsDef::XPP_USER_FS3_T0_SHIFTER, (double)41.0);
    epics.set_value(EpicsDef::XPP_USER_LAS_E0, (double)41.0);
    epics.set_value(EpicsDef::XPP_USER_LAS_E02, (double)41.0);
    epics.set_value(EpicsDef::XPP_USER_LAS_EVR0_GATE, (double)41.0);
    epics.set_value(EpicsDef::XPP_USER_LAS_EVR0_OSC, (double)41.0);
    epics.set_value(EpicsDef::XPP_USER_LAS_E_LEAK, (double)41.0);
    epics.set_value(EpicsDef::XPP_USER_LAS_E_LEAK2, (double)41.0);
    epics.set_value(EpicsDef::XPP_USER_LAS_E_PULSE, (double)41.0);
    epics.set_value(EpicsDef::XPP_USER_LAS_E_PULSE2, (double)41.0);
    epics.set_value(EpicsDef::XPP_USER_LAS_FS3_MAX, (double)41.0);
    epics.set_value(EpicsDef::XPP_USER_LAS_FS3_MIN, (double)41.0);
    epics.set_value(EpicsDef::XPP_USER_LAS_SDG0, (double)41.0);
    epics.set_value(EpicsDef::XPP_USER_LAS_T0_MONITOR, (double)41.0);
    epics.set_value(EpicsDef::XPP_USER_LAS_TIME_DELAY, (double)41.0);
    epics.set_value(EpicsDef::XPP_USER_LOM_E, (double)41.0);
    epics.set_value(EpicsDef::XPP_USER_LOM_EC, (double)41.0);
    epics.set_value(EpicsDef::XPP_USER_LXT, (double)41.0);
    epics.set_value(EpicsDef::XPP_USER_LXTTC, (double)41.0);
    epics.set_value(EpicsDef::XPP_USER_ROB_AZ, (double)41.0);
    epics.set_value(EpicsDef::XPP_USER_ROB_EL, (double)41.0);
    epics.set_value(EpicsDef::XPP_USER_ROB_J1, (double)41.0);
    epics.set_value(EpicsDef::XPP_USER_ROB_J2, (double)41.0);
    epics.set_value(EpicsDef::XPP_USER_ROB_J3, (double)41.0);
    epics.set_value(EpicsDef::XPP_USER_ROB_J4, (double)41.0);
    epics.set_value(EpicsDef::XPP_USER_ROB_J5, (double)41.0);
    epics.set_value(EpicsDef::XPP_USER_ROB_J6, (double)41.0);
    epics.set_value(EpicsDef::XPP_USER_ROB_R, (double)41.0);
    epics.set_value(EpicsDef::XPP_USER_ROB_RX, (double)41.0);
    epics.set_value(EpicsDef::XPP_USER_ROB_RY, (double)41.0);
    epics.set_value(EpicsDef::XPP_USER_ROB_RZ, (double)41.0);
    epics.set_value(EpicsDef::XPP_USER_ROB_X, (double)41.0);
    epics.set_value(EpicsDef::XPP_USER_ROB_Y, (double)41.0);
    epics.set_value(EpicsDef::XPP_USER_ROB_Z, (double)41.0);
    epics.set_value(EpicsDef::XPP_USER_VIT_TD, (double)41.0);
    epics.set_value(EpicsDef::XPP_USR_GCC_01_PMON, (double)41.0);
    epics.set_value(EpicsDef::XPP_USR_GCC_02_PMON, (double)41.0);
    epics.set_value(EpicsDef::XPP_USR_GPI_01_PMON, (double)41.0);
    epics.set_value(EpicsDef::XPP_USR_GPI_02_PMON, (double)41.0);
    epics.set_value(EpicsDef::XPP_USR_IPM_01_ChargeAmpRangeCH0, (int64_t) 42);
    epics.set_value(EpicsDef::XPP_USR_IPM_01_ChargeAmpRangeCH1, (int64_t) 42);
    epics.set_value(EpicsDef::XPP_USR_IPM_01_ChargeAmpRangeCH2, (int64_t) 42);
    epics.set_value(EpicsDef::XPP_USR_IPM_01_ChargeAmpRangeCH3, (int64_t) 42);
    epics.set_value(EpicsDef::XPP_USR_IPM_01_DiodeBias, (double)41.0);
    epics.set_value(EpicsDef::XPP_USR_LPW_01_DATA_PRI, (double)41.0);
    epics.set_value(EpicsDef::XPP_USR_LPW_01_GETAOSCALE, (int64_t) 42);
    epics.set_value(EpicsDef::XPP_USR_LPW_01_GETGAINFACTOR, (double)41.0);
    epics.set_value(EpicsDef::XPP_USR_LPW_01_GETRANGE, (double)41.0);
    epics.set_value(EpicsDef::XPP_USR_MMN_01_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_USR_MMN_02_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_USR_MMN_03_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_USR_MMN_04_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_USR_MMS_01_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_USR_MMS_02_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_USR_MMS_03_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_USR_MMS_04_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_USR_MMS_05_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_USR_MMS_06_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_USR_MMS_17_RBV, (double)41.0);
    epics.set_value(EpicsDef::XPP_USR_OXY_01_ANALOGIN, (double)41.0);
    epics.set_value(EpicsDef::XPP_USR_OXY_01_OFFSET, (double)41.0);
    epics.set_value(EpicsDef::XPP_USR_OXY_01_SCALE, (double)41.0);
    epics.set_value(EpicsDef::XPP_USR_TCT_01_GET_SOLL_1, (double)41.0);
    epics.set_value(EpicsDef::XPP_USR_TCT_01_GET_TEMP_A, (double)41.0);
    epics.set_value(EpicsDef::XPP_USR_TCT_01_GET_TEMP_B, (double)41.0);
    epics.set_value(EpicsDef::XPP_USR_TCT_01_GET_TEMP_C, (double)41.0);
    epics.set_value(EpicsDef::XPP_USR_TCT_01_GET_TEMP_D, (double)41.0);
    epics.set_value(EpicsDef::XPP_USR_TCT_01_PUT_SOLL_1, (double)41.0);
    epics.set_value(EpicsDef::XPP_USR_ao1_0, (double)41.0);
    epics.set_value(EpicsDef::XPP_USR_ao1_1, (double)41.0);
    epics.set_value(EpicsDef::XPP_USR_ao1_10, (double)41.0);
    epics.set_value(EpicsDef::XPP_USR_ao1_11, (double)41.0);
    epics.set_value(EpicsDef::XPP_USR_ao1_12, (double)41.0);
    epics.set_value(EpicsDef::XPP_USR_ao1_13, (double)41.0);
    epics.set_value(EpicsDef::XPP_USR_ao1_14, (double)41.0);
    epics.set_value(EpicsDef::XPP_USR_ao1_15, (double)41.0);
    epics.set_value(EpicsDef::XPP_USR_ao1_2, (double)41.0);
    epics.set_value(EpicsDef::XPP_USR_ao1_3, (double)41.0);
    epics.set_value(EpicsDef::XPP_USR_ao1_4, (double)41.0);
    epics.set_value(EpicsDef::XPP_USR_ao1_5, (double)41.0);
    epics.set_value(EpicsDef::XPP_USR_ao1_6, (double)41.0);
    epics.set_value(EpicsDef::XPP_USR_ao1_7, (double)41.0);
    epics.set_value(EpicsDef::XPP_USR_ao1_8, (double)41.0);
    epics.set_value(EpicsDef::XPP_USR_ao1_9, (double)41.0);
#else // defined(EPICSSTREAMID_UNITTEST)
    epics.set_value(EpicsDef::XPP_VARS_FLOAT_02, (double)41.0);
    epics.set_value(EpicsDef::XPP_VARS_FLOAT_03, (double)41.0);
    epics.set_value(EpicsDef::XPP_VARS_FLOAT_04, (double)41.0);
    epics.set_value(EpicsDef::XPP_VARS_FLOAT_05, (double)41.0);
    epics.set_value(EpicsDef::XPP_VARS_FLOAT_06, (double)41.0);
    epics.set_value(EpicsDef::XPP_VARS_FLOAT_07, (double)41.0);
    epics.set_value(EpicsDef::XPP_VARS_FLOAT_08, (double)41.0);
    epics.set_value(EpicsDef::XPP_VARS_FLOAT_09, (double)41.0);
    epics.set_value(EpicsDef::XPP_VARS_FLOAT_10, (double)41.0);
    epics.set_string(EpicsDef::XPP_VARS_STRING_01, "Test String");
    epics.set_string(EpicsDef::XPP_VARS_STRING_02, "Test String");
    epics.set_string(EpicsDef::XPP_VARS_STRING_03, "Test String");
    epics.set_string(EpicsDef::XPP_VARS_STRING_04, "Test String");
    epics.set_string(EpicsDef::XPP_VARS_STRING_05, "Test String");
    epics.set_string(EpicsDef::XPP_VARS_STRING_06, "Test String");
    epics.set_string(EpicsDef::XPP_VARS_STRING_07, "Test String");
    epics.set_string(EpicsDef::XPP_VARS_STRING_08, "Test String");
    epics.set_string(EpicsDef::XPP_VARS_STRING_09, "Test String");
#endif
    
}
   

void addNames(Xtc& xtc, NamesLookup& namesLookup, unsigned& nodeId, unsigned segment) {
    Alg xppEpicsAlg("fuzzy",0,0,0);
    NamesId namesId1(nodeId,1+10*segment);
    Names& epicsNames = *new(xtc) Names("xppepics", xppEpicsAlg, "epics","detnum1234", namesId1, segment);
    epicsNames.add(xtc, EpicsDef);
    namesLookup[namesId1] = NameIndex(epicsNames);

}

void addData(Xtc& xtc, NamesLookup& namesLookup, unsigned nodeId, unsigned segment) {
    NamesId namesId1(nodeId,1+10*segment);
    epicsExample(xtc, namesLookup, namesId1);
}

void usage(char* progname)
{
    fprintf(stderr, "Usage: %s [-f <filename> -n <numEvents> -t -h]\n", progname);
}

Dgram& createTransition(TransitionId::Value transId) {
    TypeId tid(TypeId::Parent, 0);
    uint64_t pulseId = 0xffff;
    uint32_t env = 0;
    struct timeval tv;
    void* buf = malloc(BUFSIZE);
    gettimeofday(&tv, NULL);
    Sequence seq(Sequence::Event, transId, TimeStamp(tv.tv_sec, tv.tv_usec), PulseId(pulseId,0));
    return *new(buf) Dgram(Transition(seq, env), Xtc(tid));
}

void save(Dgram& dg, FILE* xtcFile) {
    if (fwrite(&dg, sizeof(dg) + dg.xtc.sizeofPayload(), 1, xtcFile) != 1) {
        printf("Error writing to output xtc file.\n");
    }
}

#define MAX_FNAME_LEN 256

int main(int argc, char* argv[])
{
    int c;
    int parseErr = 0;
    unsigned nevents = 2;
    char xtcname[MAX_FNAME_LEN];
    char* tsname = 0;
    char* bdXtcname = 0; // Bigdata xtc - if specified, timestamps are from this file
    strncpy(xtcname, "epics.xtc2", MAX_FNAME_LEN); // Fixed output filename

    while ((c = getopt(argc, argv, "ht:f:n:")) != -1) {
        switch (c) {
            case 'h':
                usage(argv[0]);
                exit(0);
            case 't':
                tsname = optarg;
                break;
            case 'n':
                nevents = atoi(optarg);
                break;
            case 'f':
                bdXtcname = optarg;
                break;
            default:
                parseErr++;
        }
    }

    // Output xtc file
    FILE* xtcFile = fopen(xtcname, "w");
    if (!xtcFile) {
        printf("Error opening output xtc file.\n");
        return -1;
    }

    // Bigdata xtc file
    int fd = -1;
    if (bdXtcname != 0) {
        fd = open(bdXtcname, O_RDONLY);
        if (fd < 0) {
            fprintf(stderr, "Unable to open bigdata xtc file %s\n", bdXtcname);
            exit(2);
        }
    }

    XtcFileIterator dgInIter(fd, BUFSIZE);
    Dgram* dgIn;

    // Read timestamp 
    array<time_t, 500> sec_arr = {};
    array<long, 500> nsec_arr = {};
    sec_arr[0] = 0;
    nsec_arr[0] = 0;
    if (tsname != 0) {
        ifstream tsfile(tsname);
        time_t sec = 0;
        long nsec = 0;
        int i = 0;
        while (tsfile >> sec >> nsec) {
            sec_arr[i] = sec;
            nsec_arr[i] = nsec;
            cout << i << ":" << sec_arr[i] << " " << nsec_arr[i] << endl;
            i++;
        }  
        printf("found %d timestamps\n", i); 
    }

    struct timeval tv;
    TypeId tid(TypeId::Parent, 0);
    uint32_t env = 0;
    uint64_t pulseId = 0;

    Dgram& config = createTransition(TransitionId::Configure);

    unsigned nodeid1 = 1;
    NamesLookup namesLookup1;
    unsigned nSegments=1;
    for (unsigned iseg=0; iseg<nSegments; iseg++) {
        addNames(config.xtc, namesLookup1, nodeid1, iseg);
        addData(config.xtc, namesLookup1, nodeid1, iseg);
    }

    save(config,xtcFile);

    DebugIter iter(&config.xtc, namesLookup1);
    iter.iterate();

    void* buf = malloc(BUFSIZE);
    for (int i = 0; i < nevents; i++) {
        // Determine how which timestamps to be used: from bigdata xtc,
        // from text file (space separated), or current time.
        Sequence seq;
        if (bdXtcname != 0) {
            dgIn = dgInIter.next();
            seq = dgIn->seq;
        } else if (tsname != 0) {
            seq = Sequence(Sequence::Event, TransitionId::L1Accept, TimeStamp(sec_arr[i], nsec_arr[i]), PulseId(pulseId, 0));
            cout << "Timestamp from file: " << sec_arr[i] << " " << nsec_arr[i] << endl;
        } else {
            gettimeofday(&tv, NULL);
            seq = Sequence(Sequence::Event, TransitionId::L1Accept, TimeStamp(tv.tv_sec, tv.tv_usec), PulseId(pulseId,0));
        }

        Dgram& dgram = *new(buf) Dgram(Transition(seq, env), Xtc(tid));

        for (unsigned iseg=0; iseg<nSegments; iseg++) {
            addData(dgram.xtc, namesLookup1, nodeid1, iseg);
        }

        DebugIter iter(&dgram.xtc, namesLookup1);
        iter.iterate();

        save(dgram,xtcFile);
     }

    fclose(xtcFile);

    return 0;
}
