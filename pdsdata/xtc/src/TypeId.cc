#include "pdsdata/xtc/TypeId.hh"

#include <stdlib.h>
#include <string.h>

using namespace Pds;

TypeId::TypeId(Type type, uint32_t version, bool cmp) :
  _value((version<<16 )| type | (cmp ? 0x80000000:0)) {}

TypeId::TypeId(const char* s) :
  _value(NumberOf)
{
  const char* token = strrchr(s,'_');
  if (!(token && *(token+1)=='v')) return;

  char* e;
  unsigned vsn = strtoul(token+2,&e,10);
  if (e==token+2 || *e!=0) return;

  char* p = strndup(s,token-s);
  for(unsigned i=0; i<NumberOf; i++)
    if (strcmp(p,name((Type)i))==0)
      _value = (vsn<<16) | i;
  free(p);
}

TypeId::TypeId(const TypeId& v) : _value(v._value) {}

uint32_t TypeId::value() const {return _value;}

uint32_t TypeId::version() const {return (_value&0xffff0000)>>16;}

TypeId::Type TypeId::id() const {return (TypeId::Type)(_value&0xffff);}

bool     TypeId::compressed() const { return _value&0x80000000; }

unsigned TypeId::compressed_version() const { return (_value&0x7fff0000)>>16; }

bool     TypeId::is_configuration() const
{
  static Type _configuration_types[] =
    { Id_AcqConfig,
      Id_Opal1kConfig,
      Id_FrameFexConfig,
      Id_TM6740Config,
      Id_ControlConfig,
      Id_pnCCDconfig,
      Id_PrincetonConfig,
      Id_FrameFccdConfig,
      Id_FccdConfig,
      Id_IpimbConfig,
      Id_EncoderConfig,
      Id_EvrIOConfig,
      Id_CspadConfig,
      Id_IpmFexConfig,  // LUSI Diagnostics
      Id_DiodeFexConfig,
      Id_PimImageConfig,
      Id_AcqTdcConfig,
      Id_XampsConfig,
      Id_Cspad2x2Config,
      Id_FexampConfig,
      Id_Gsc16aiConfig,
      Id_PhasicsConfig,
      Id_TimepixConfig,
      Id_OceanOpticsConfig,
      Id_FliConfig,
      Id_QuartzConfig,
      Id_AndorConfig,
      Id_UsdUsbConfig,
      Id_OrcaConfig,
      Id_ImpConfig,
      Id_AliasConfig,
      Id_L3TConfig,
      Id_RayonixConfig,
      Id_EpixConfig,
      Id_EpixSamplerConfig,
      Id_Epix10kConfig,
      Id_Epix100aConfig,
      Id_EvsConfig,
      Id_PartitionConfig,
      Id_PimaxConfig,
      Id_GenericPgpConfig,
      Id_TimeToolConfig,
      Id_EpixSConfig,
      Id_GotthardConfig,
      Id_Andor3dConfig,
      Id_Generic1DConfig,
      Id_UsdUsbFexConfig,
      Id_ControlsCameraConfig,
      Id_ArchonConfig,
      Id_JungfrauConfig,
      Id_QuadAdcConfig,
      Id_ZylaConfig,
    };
  const unsigned nconfigtypes = sizeof(_configuration_types)/sizeof(Type);
  Type t = id();
  for(unsigned i=0; i<nconfigtypes; i++)
    if (t == _configuration_types[i])
      return true;
  return false;
}

const char* TypeId::name(Type type)
{
   static const char* _names[NumberOf] = {
    "Any",                     // 0
    "Xtc",                     // 1
    "Frame",                   // 2
    "AcqWaveform",             // 3
    "AcqConfig",               // 4
    "TwoDGaussian",            // 5
    "Opal1kConfig",            // 6
    "FrameFexConfig",          // 7
    "EvrConfig",               // 8
    "TM6740Config",            // 9
    "RunControlConfig",        // 10
    "pnCCDframe",              // 11
    "pnCCDconfig",             // 12
    "Epics",                   // 13
    "FEEGasDetEnergy",         // 14
    "EBeamBld",                // 15
    "PhaseCavity",             // 16
    "PrincetonFrame",          // 17
    "PrincetonConfig",         // 18
    "EvrData",                 // 19
    "FrameFccdConfig",         // 20
    "FccdConfig",              // 21
    "IpimbData",               // 22
    "IpimbConfig",             // 23
    "EncoderData",             // 24
    "EncoderConfig",           // 25
    "EvrIOConfig",             // 26
    "PrincetonInfo",           // 27
    "CspadElement",            // 28
    "CspadConfig",             // 29
    "IpmFexConfig",            // 30
    "IpmFex",                  // 31
    "DiodeFexConfig",          // 32
    "DiodeFex",                // 33
    "PimImageConfig",          // 34
    "SharedIpimb",             // 35
    "AcqTDCConfig",            // 36
    "AcqTDCData",              // 37
    "Index",                   // 38
    "XampsConfig",             // 39
    "XampsElement",            // 40
    "Cspad2x2Element",         // 41
    "SharedPIM",               // 42
    "Cspad2x2Config",          // 43
    "FexampConfig",            // 44
    "FexampElement",           // 45
    "Gsc16aiConfig",           // 46
    "Gsc16aiData",             // 47
    "PhasicsConfig",           // 48
    "TimepixConfig",           // 49
    "TimepixData",             // 50
    "CspadCompressedElement",  // 51
    "OceanOpticsConfig",       // 52
    "OceanOpticsData",         // 53
    "EpicsConfig",             // 54
    "FliConfig",               // 55
    "FliFrame",                // 56
    "QuartzConfig",            // 57
    "Reserved1",               // 58
    "Reserved2",               // 59
    "AndorConfig",             // 60
    "AndorFrame",              // 61
    "UsdUsbData",              // 62
    "UsdUsbConfig",            // 63
    "GMD",                     // 64
    "SharedAcqADC",            // 65
    "OrcaConfig",              // 66
    "ImpData",                 // 67
    "ImpConfig",               // 68
    "AliasConfig",             // 69
    "L3TConfig",               // 70
    "L3TData",                 // 71
    "Spectrometer",            // 72
    "RayonixConfig",           // 73
    "EpixConfig",              // 74
    "EpixElement",             // 75
    "EpixSamplerConfig",       // 76
    "EpixSamplerElement",      // 77
    "EvsConfig",               // 78
    "PartitionConfig",         // 79
    "PimaxConfig",             // 80
    "PimaxFrame",              // 81
    "Arraychar",               // 82
    "Epix10kConfig",           // 83
    "Epix100aConfig",          // 84
    "GenericPgpConfig",        // 85
    "TimeToolConfig",          // 86
    "TimeToolData",            // 87
    "EpixSConfig",             // 88
    "SmlDataConfig",           // 89
    "SmlDataOrigDgramOffset",  // 90
    "SmlDataProxy",            // 91
    "ArrayUInt16",             // 92
    "GotthardConfig",          // 93
    "AnalogInput",             // 94
    "SmlData",                 // 95
    "Andor3dConfig",           // 96
    "Andor3dFrame",            // 97
    "BeamMonitorBldData",      // 98
    "Generic1DConfig",         // 99
    "Generic1DData",           // 100
    "UsdUsbFexConfig",         // 101
    "UsdUsbFexData",           // 102
    "EOrbits",                 // 103
    "SharedUsdUsb",            // 104
    "ControlsCameraConfig",    // 105
    "ArchonConfig",            // 106
    "JungfrauConfig",          // 107
    "JungfrauElement",         // 108
    "QuadAdcConfig",           // 109
    "ZylaConfig",              // 110
    "ZylaFrame",               // 111
  };
   const char* p = (type < NumberOf ? _names[type] : "-Invalid-");
   if (!p) p = "-Unnamed-";
   return p;
}
