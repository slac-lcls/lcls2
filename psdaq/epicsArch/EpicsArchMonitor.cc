#include <stdio.h>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "psdaq/service/GenericPool.hh"
#include "psalg/utils/SysLog.hh"

#include "EpicsArchMonitor.hh"

using logging = psalg::SysLog;

namespace Pds
{

using std::string;
using std::stringstream;

EpicsArchMonitor::EpicsArchMonitor(const std::string& sFnConfig,
                                   int iDebugLevel,
                                   std::string& sConfigFileWarning):
  _sFnConfig(sFnConfig),
  _iDebugLevel(iDebugLevel),
  _epicsArchDef()
{
  if (_sFnConfig == "") {
    string msg("EpicsArchMonitor::EpicsArchMonitor(): Invalid parameters");
    logging::critical("%s", msg.c_str());
    throw msg;
  }

  Pds::PvConfigFile::TPvList vPvList;
  int iMaxDepth = 10;
  bool bProviderType = false;
  PvConfigFile configFile(_sFnConfig, bProviderType, iMaxDepth, iMaxNumPv, (_iDebugLevel >= 1));
  int iFail = configFile.read(vPvList, sConfigFileWarning);
  if (iFail != 0) {
    string msg("EpicsArchMonitor::EpicsArchMonitor(): configFile(" + _sFnConfig + ").read() failed\n");
    logging::critical("%s", msg.c_str());
    throw msg;
  }

  if (vPvList.empty()) {
    string msg(
      "EpicsArchMonitor::EpicsArchMonitor(): No Pv Name is specified in the config file "
      + _sFnConfig);
    logging::critical("%s", msg.c_str());
    throw msg;
  }

  logging::debug("Monitoring PV:");
  for (unsigned iPv = 0; iPv < vPvList.size(); iPv++)
    logging::debug("  [%d] %-32s PV %-32s Provider %s", iPv,
           vPvList[iPv].sPvDescription.c_str(), vPvList[iPv].sPvName.c_str(),
           vPvList[iPv].bProviderType ? "PVA" : "CA");

  iFail = _setupPvList(vPvList, _lpvPvList);
  if (iFail != 0) {
    string msg("EpicsArchMonitor::EpicsArchMonitor()::setupPvList() Failed");
    logging::critical("%s", msg.c_str());
    throw msg;
  }
}

EpicsArchMonitor::~EpicsArchMonitor()
{
  for (unsigned iPvName = 0; iPvName < _lpvPvList.size(); iPvName++)
  {
    EpicsMonitorPv& epicsPvCur = *_lpvPvList[iPvName];
    int iFail = epicsPvCur.release();

    if (iFail != 0)
      logging::error("EpicsArchMonitor()::EpicsMonitorPv::release(%s (%s)) failed",
                     epicsPvCur.getPvDescription().c_str(), epicsPvCur.getPvName().c_str());
  }
  EpicsMonitorPv::shutdown();
}

void EpicsArchMonitor::close()
{
  EpicsMonitorPv::close();
}

void EpicsArchMonitor::initDef(size_t& payloadSize)
{
  payloadSize = 0;
  for (unsigned iPvName = 0; iPvName < _lpvPvList.size(); iPvName++)
  {
    EpicsMonitorPv& epicsPvCur = *_lpvPvList[iPvName];
    size_t size;
    epicsPvCur.addDef(_epicsArchDef, size);
    payloadSize += size;
  }
}

void EpicsArchMonitor::addNames(const std::string& detName, const std::string& detType, const std::string& serNo,
                                XtcData::Xtc& xtc, XtcData::NamesLookup& namesLookup, unsigned nodeId)
{
  XtcData::Alg alg("raw", 2, 0, 0);
  XtcData::NamesId namesId(nodeId, iNamesIndex);
  XtcData::Names& names = *new(xtc) XtcData::Names(detName.c_str(), alg,
                                                   detType.c_str(), serNo.c_str(), namesId);
  names.add(xtc, _epicsArchDef);
  namesLookup[namesId] = XtcData::NameIndex(names);
}

int EpicsArchMonitor::_writeToXtc(XtcData::Xtc& xtc, XtcData::NamesLookup& namesLookup, XtcData::NamesId& namesId)
{
  XtcData::DescribedData desc(xtc, namesLookup, namesId);

  const size_t iNumPv = _lpvPvList.size();
  std::vector<size_t> length(iNumPv);
  uint32_t* staleFlags = static_cast<uint32_t*>(desc.data());
  unsigned nWords = 1 + ((iNumPv - 1) >> 5);
  memset(staleFlags, 0, nWords * sizeof(*staleFlags));
  char* pXtc = reinterpret_cast<char*>(&staleFlags[nWords]);
  for (unsigned iPvName = 0; iPvName < iNumPv; iPvName++)
  {
    EpicsMonitorPv& epicsPvCur = *_lpvPvList[iPvName];

    if (_iDebugLevel >= 1)
      epicsPvCur.printPv();

    size_t size = 0;
    bool stale;
    epicsPvCur.addToXtc(stale, pXtc, size, length[iPvName]);
    if (stale) staleFlags[iPvName >> 5] |= 1 << (iPvName & 0x1f);

    pXtc += size;
  }

  desc.set_data_length(pXtc - reinterpret_cast<char*>(&xtc));

  unsigned shape[XtcData::MaxRank];
  shape[0] = unsigned(nWords);
  desc.set_array_shape(EpicsArchDef::Stale, shape);
  for (unsigned iPvName = 0; iPvName < iNumPv; iPvName++)
  {
    shape[0] = unsigned(length[iPvName]);
    desc.set_array_shape(EpicsArchDef::Data + iPvName, shape);
  }

  return 0;     // All PV values are outputted successfully
}

void EpicsArchMonitor::getData(XtcData::Xtc& xtc, XtcData::NamesLookup& namesLookup, unsigned nodeId)
{
  XtcData::NamesId namesId(nodeId, iNamesIndex);
  _writeToXtc(xtc, namesLookup, namesId);
}

unsigned EpicsArchMonitor::validate(unsigned& iPvCount)
{
  const size_t iNumPv = _lpvPvList.size();
  iPvCount = iNumPv;

  unsigned nNotConnected = 0;
  for (unsigned iPvName = 0; iPvName < iNumPv; iPvName++)
  {
    EpicsMonitorPv& epicsPvCur = *_lpvPvList[iPvName];

    // Take all fields and reject uninteresting ones in EpicsMonitorPv::addDef()
    const std::string request("field()");

    // Revisit: Test ready here?  It blocks with timeout
    bool ready = epicsPvCur.ready(request);
    if (!ready || !epicsPvCur.isConnected()) {
      //epicsPvCur.reconnect();
      printf("%s (%s) is not %s\n",
             epicsPvCur.getPvDescription().c_str(), epicsPvCur.getPvName().c_str(),
             ready ? "ready" : "connected");
      nNotConnected++;
    }
  }

  return nNotConnected;
}

int EpicsArchMonitor::_setupPvList(const Pds::PvConfigFile::TPvList& vPvList,
                                   TEpicsMonitorPvList& lpvPvList)
{
  if (vPvList.empty())
    return 0;
  if (vPvList.size() >= iMaxNumPv)
    printf("EpicsArchMonitor::_setupPvList(): Number of PVs (%zd) has reached capacity (%d), "
           "some PVs in the list were skipped.\n", vPvList.size(), iMaxNumPv);

  if (EpicsMonitorPv::prepare())
    return 1;

  for (unsigned iPvName = 0; iPvName < vPvList.size(); iPvName++)
  {
    std::shared_ptr<EpicsMonitorPv> epicsPvCur =
      std::make_shared<EpicsMonitorPv>(vPvList[iPvName].sPvName,
                                       vPvList[iPvName].sPvDescription,
                                       vPvList[iPvName].bProviderType);
    lpvPvList.push_back(epicsPvCur);
  }
  return 0;
}

}       // namespace Pds
