#include <stdio.h>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/Damage.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "psdaq/service/GenericPool.hh"
#include "psalg/utils/SysLog.hh"

#include "EpicsArchMonitor.hh"

using logging = psalg::SysLog;

namespace Drp
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

  PvConfigFile::TPvList vPvList;
  int iMaxDepth = 10;
  std::string sProvider = "ca";
  PvConfigFile configFile(_sFnConfig, sProvider, iMaxDepth, iMaxNumPv, (_iDebugLevel >= 1));
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
    logging::debug("  [%d] %-32s PV %-32s Provider '%s'", iPv,
                   vPvList[iPv].sPvDescription.c_str(),
                   vPvList[iPv].sPvName.c_str(),
                   vPvList[iPv].sProvider.c_str());

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
}

void EpicsArchMonitor::close()
{
  EpicsMonitorPv::close();
}

void EpicsArchMonitor::_initDef()
{
  for (unsigned iPvName = 0; iPvName < _lpvPvList.size(); iPvName++)
  {
    EpicsMonitorPv& epicsPvCur = *_lpvPvList[iPvName];
    epicsPvCur.addDef(_epicsArchDef);
  }
}

void EpicsArchMonitor::_initInfoDef()
{
  _epicsInfoDef.NameVec.push_back({"keys", XtcData::Name::CHARSTR, 1});

  for (unsigned iPvName = 0; iPvName < _lpvPvList.size(); iPvName++)
  {
    EpicsMonitorPv& epicsPvCur = *_lpvPvList[iPvName];

    auto& descr   = epicsPvCur.getPvDescription();
    auto  detName = !descr.empty() ? descr : epicsPvCur.getPvName();
    _epicsInfoDef.NameVec.push_back({detName.c_str(), XtcData::Name::CHARSTR, 1});
  }
}

void EpicsArchMonitor::_addInfo(XtcData::CreateData& epicsInfo)
{
  // add dictionary of information for each epics detname above.
  // first name is required to be "keys".  keys and values
  // are delimited by ",".
  epicsInfo.set_string(0, "epicsname" "," "alias");

  for (unsigned iPvName = 0; iPvName < _lpvPvList.size(); iPvName++)
  {
    EpicsMonitorPv& epicsPvCur = *_lpvPvList[iPvName];

    epicsInfo.set_string(1 + iPvName, (epicsPvCur.getPvName() + "," + epicsPvCur.getPvDescription()).c_str());
  }
}

void EpicsArchMonitor::addNames(const std::string& detName, const std::string& detType, const std::string& serNo,
                                XtcData::Xtc& xtc, XtcData::NamesLookup& namesLookup, unsigned nodeId)
{
  XtcData::Alg     rawAlg("raw", 2, 0, 0);
  XtcData::NamesId rawNamesId(nodeId, iRawNamesIndex);
  XtcData::Names&  rawNames = *new(xtc) XtcData::Names(detName.c_str(), rawAlg,
                                                       detType.c_str(), serNo.c_str(), rawNamesId);
  _initDef();
  rawNames.add(xtc, _epicsArchDef);
  namesLookup[rawNamesId] = XtcData::NameIndex(rawNames);

  XtcData::Alg     infoAlg("epicsinfo", 1, 0, 0);
  XtcData::NamesId infoNamesId(nodeId, iInfoNamesIndex);
  XtcData::Names&  infoNames = *new(xtc) XtcData::Names("epicsinfo", infoAlg,
                                                        "epicsinfo", "detnum1234", infoNamesId);
  _initInfoDef();
  infoNames.add(xtc, _epicsInfoDef);
  namesLookup[infoNamesId] = XtcData::NameIndex(infoNames);

  XtcData::CreateData epicsInfo(xtc, namesLookup, infoNamesId);
  _addInfo(epicsInfo);
}

int EpicsArchMonitor::getData(XtcData::Xtc& xtc, XtcData::NamesLookup& namesLookup, unsigned nodeId, size_t payloadSize)
{
  XtcData::NamesId namesId(nodeId, iRawNamesIndex);
  XtcData::DescribedData desc(xtc, namesLookup, namesId);
  payloadSize -= xtc.sizeofPayload();
  auto scootch = 64;             // Size dependent amount used by DescribedData
  payloadSize -= scootch;

  const size_t iNumPv = _lpvPvList.size();
  std::vector<std::vector<unsigned> > shapes(iNumPv);
  uint32_t* staleFlags = static_cast<uint32_t*>(desc.data());
  unsigned nWords = 1 + ((iNumPv - 1) >> 5);
  memset(staleFlags, 0, nWords * sizeof(*staleFlags));
  char* pXtc = reinterpret_cast<char*>(&staleFlags[nWords]);
  payloadSize -= pXtc - reinterpret_cast<char*>(&xtc);
  for (unsigned iPvName = 0; iPvName < iNumPv; iPvName++)
  {
    EpicsMonitorPv& epicsPvCur = *_lpvPvList[iPvName];

    if (_iDebugLevel >= 1)
      epicsPvCur.printPv();

    size_t size = payloadSize;
    bool stale;
    if (!epicsPvCur.addToXtc(stale, pXtc, size, shapes[iPvName]))
    {
      if (size > payloadSize) {
        logging::debug("Truncated: Buffer of size %zu is too small for payload of size %zu for %s\n",
                       payloadSize, size, epicsPvCur.name().c_str());
        xtc.damage.increase(XtcData::Damage::Truncated);
        size = payloadSize;
      }
      if (stale) staleFlags[iPvName >> 5] |= 1 << (iPvName & 0x1f);

      pXtc        += size;
      payloadSize -= size;
    }
  }

  desc.set_data_length(pXtc - reinterpret_cast<char*>(&xtc));

  unsigned shape[XtcData::MaxRank];
  shape[0] = unsigned(nWords);
  desc.set_array_shape(EpicsArchDef::Stale, shape);
  for (unsigned iPvName = 0; iPvName < iNumPv; iPvName++)
  {
    EpicsMonitorPv& epicsPvCur = *_lpvPvList[iPvName];
    if (!epicsPvCur.isDisabled())
      desc.set_array_shape(EpicsArchDef::Data + iPvName, shapes[iPvName].data());
  }

  return 0;     // All PV values are outputted successfully
}

unsigned EpicsArchMonitor::validate(unsigned& iPvCount)
{
  const size_t iNumPv = _lpvPvList.size();
  iPvCount = iNumPv;

  unsigned nNotConnected = 0;
  for (unsigned iPvName = 0; iPvName < iNumPv; iPvName++)
  {
    EpicsMonitorPv& epicsPvCur = *_lpvPvList[iPvName];

    // Take select fields of interest and ignore uninteresting ones
    const std::string request("field(value,timeStamp,dimension)");

    // Revisit: Test ready here?  It blocks with timeout
    unsigned tmo = 3;
    bool ready = epicsPvCur.ready(request, tmo);
    if (!ready || !epicsPvCur.isConnected()) {
      epicsPvCur.disable();
      printf("%s (%s) is not %s\n",
             epicsPvCur.getPvDescription().c_str(), epicsPvCur.getPvName().c_str(),
             ready ? "ready" : "connected");
      nNotConnected++;
    }
  }

  return nNotConnected;
}

int EpicsArchMonitor::_setupPvList(const PvConfigFile::TPvList& vPvList,
                                   TEpicsMonitorPvList& lpvPvList)
{
  if (vPvList.empty())
    return 0;
  if (vPvList.size() >= iMaxNumPv)
    printf("EpicsArchMonitor::_setupPvList(): Number of PVs (%zd) has reached capacity (%d), "
           "some PVs in the list were skipped.\n", vPvList.size(), iMaxNumPv);

  for (unsigned iPvName = 0; iPvName < vPvList.size(); iPvName++)
  {
    std::shared_ptr<EpicsMonitorPv> epicsPvCur =
      std::make_shared<EpicsMonitorPv>(vPvList[iPvName].sPvName,
                                       vPvList[iPvName].sPvDescription,
                                       vPvList[iPvName].sProvider,
                                       _iDebugLevel != 0);
    lpvPvList.push_back(epicsPvCur);
  }
  return 0;
}

}       // namespace Drp
