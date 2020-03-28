#ifndef EPICS_ARCH_MONITOR_HH
#define EPICS_ARCH_MONITOR_HH

#include <string>
#include <vector>
#include <set>
#include "xtcdata/xtc/VarDef.hh"
#include "EpicsXtcSettings.hh"
#include "EpicsMonitorPv.hh"
#include "PvConfigFile.hh"

namespace XtcData
{
  class Dgram;
};

namespace Pds
{
  class UserMessage;

  class EpicsArchMonitor
  {
  public:
    EpicsArchMonitor(const std::string& sFnConfig, int iDebugLevel,
                     std::string& sConfigFileWarning);
    ~EpicsArchMonitor();
  public:
    void     initDef(size_t& payloadSize);
    void     addNames(const std::string& detName, const std::string& detType, const std::string& serNo,
                      XtcData::Xtc& xtc, XtcData::NamesLookup& namesLookup, unsigned nodeId);
    void     getData(XtcData::Xtc& xtc, XtcData::NamesLookup& namesLookup, unsigned nodeId);
    unsigned validate(unsigned& iPvCount);

    static void close();

    static const int  iNamesIndex   = EpicsXtcSettings::iNamesIndex;
    static const int  iMaxNumPv     = EpicsXtcSettings::iMaxNumPv;
    static const int  iMaxXtcSize   = EpicsXtcSettings::iMaxXtcSize;

  private:
    std::string         _sFnConfig;
    int                 _iDebugLevel;
    TEpicsMonitorPvList _lpvPvList;
    EpicsArchDef        _epicsArchDef;

    int _setupPvList(const Pds::PvConfigFile::TPvList& vPvList, TEpicsMonitorPvList& lpvPvList);
    int _writeToXtc(XtcData::Xtc& xtc, XtcData::NamesLookup& namesLookup, XtcData::NamesId& namesId);

    // Class usage control: Value semantics is disabled
    EpicsArchMonitor(const EpicsArchMonitor&) = delete;
    EpicsArchMonitor& operator=(const EpicsArchMonitor&) = delete;
  };

}       // namespace Pds

#endif
