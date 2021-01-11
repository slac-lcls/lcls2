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

namespace Drp
{
  class UserMessage;

  class EpicsArchMonitor
  {
  public:
    EpicsArchMonitor(const std::string& sFnConfig, int iDebugLevel,
                     std::string& sConfigFileWarning);
    ~EpicsArchMonitor();
  public:
    void     addNames(const std::string& detName, const std::string& detType, const std::string& serNo,
                      XtcData::Xtc& xtc, XtcData::NamesLookup& namesLookup, unsigned nodeId,
                      size_t& payloadSize);
    int      getData(XtcData::Xtc& xtc, XtcData::NamesLookup& namesLookup, unsigned nodeId, size_t payloadSize, uint64_t& nStales);
    unsigned validate(unsigned& iPvCount, unsigned tmo);
    unsigned validate(unsigned& iPvCount);

    static void close();

    static const int  iRawNamesIndex  = EpicsXtcSettings::iRawNamesIndex;
    static const int  iInfoNamesIndex = EpicsXtcSettings::iInfoNamesIndex;
    static const int  iMaxNumPv       = EpicsXtcSettings::iMaxNumPv;
    static const int  iMaxXtcSize     = EpicsXtcSettings::iMaxXtcSize;

  private:
    std::string         _sFnConfig;
    int                 _iDebugLevel;
    TEpicsMonitorPvList _lpvPvList;
    EpicsArchDef        _epicsArchDef;
    XtcData::VarDef     _epicsInfoDef;

    void _initDef(size_t& payloadSize);
    void _initInfoDef();
    void _addInfo(XtcData::CreateData& epicsInfo);
    int  _setupPvList(const PvConfigFile::TPvList& vPvList, TEpicsMonitorPvList& lpvPvList);

    // Class usage control: Value semantics is disabled
    EpicsArchMonitor(const EpicsArchMonitor&) = delete;
    EpicsArchMonitor& operator=(const EpicsArchMonitor&) = delete;
  };

}       // namespace Drp

#endif
