#ifdef NDEBUG
#undef NDEBUG
#endif

#include <stdio.h>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/Damage.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "psdaq/service/EbDgram.hh"
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

void EpicsArchMonitor::_initDef(size_t& payloadSize)
{
  _epicsArchDef.NameVec.clear();
  _epicsArchDef.NameVec.push_back({"StaleFlags", XtcData::Name::UINT32, 1});

  payloadSize = 0;
  for (unsigned iPvName = 0; iPvName < _lpvPvList.size(); iPvName++)
  {
    EpicsMonitorPv& epicsPvCur = *_lpvPvList[iPvName];
    size_t size;
    if (epicsPvCur.addDef(_epicsArchDef, size))
      logging::warning("addDef failed for %s", epicsPvCur.getPvName().c_str());
    payloadSize += size;
  }
}

void EpicsArchMonitor::_initInfoDef()
{
  _epicsInfoDef.NameVec.clear();
  _epicsInfoDef.NameVec.push_back({"keys", XtcData::Name::CHARSTR, 1});

  for (unsigned iPvName = 0; iPvName < _lpvPvList.size(); iPvName++)
  {
    EpicsMonitorPv& epicsPvCur = *_lpvPvList[iPvName];
    if (!epicsPvCur.isDisabled())
    {
      auto& descr   = epicsPvCur.getPvDescription();
      auto  detName = !descr.empty() ? descr : epicsPvCur.getPvName();
      _epicsInfoDef.NameVec.push_back({detName.c_str(), XtcData::Name::CHARSTR, 1});
    }
  }
}

void EpicsArchMonitor::_addInfo(XtcData::CreateData& epicsInfo)
{
  // add dictionary of information for each epics detname above.
  // first name is required to be "keys".  keys and values
  // are delimited by ",".
  unsigned index = 0;
  epicsInfo.set_string(index++, "epicsname"); // "," "2nd string"...

  for (unsigned iPvName = 0; iPvName < _lpvPvList.size(); iPvName++)
  {
    EpicsMonitorPv& epicsPvCur = *_lpvPvList[iPvName];
    if (!epicsPvCur.isDisabled())
    {
      epicsInfo.set_string(index++, epicsPvCur.getPvName().c_str()); // + "," + 2ndString).c_str()
    }
  }
}

void EpicsArchMonitor::addNames(const std::string& detName, const std::string& detType, const std::string& serNo,
                                XtcData::Xtc& xtc, XtcData::NamesLookup& namesLookup, unsigned nodeId,
                                size_t& payloadSize)
{
  XtcData::Alg     rawAlg("raw", 2, 0, 0);
  XtcData::NamesId rawNamesId(nodeId, iRawNamesIndex);
  XtcData::Names&  rawNames = *new(xtc) XtcData::Names(detName.c_str(), rawAlg,
                                                       detType.c_str(), serNo.c_str(), rawNamesId);
  _initDef(payloadSize);
  payloadSize += (sizeof(Pds::EbDgram)    + // An EbDgram is needed by the MEB
                  24                      + // Space needed by DescribedData
                  sizeof(XtcData::Shapes) + // Needed by DescribedData
                  sizeof(XtcData::Shape)  + // 1 for the stale vector
                  sizeof(XtcData::Shape) * _lpvPvList.size()); // 1 per PV
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
  payloadSize -= xtc.sizeofPayload();     // = the '24' in addNames()
  payloadSize -= sizeof(XtcData::Shapes); // Reserve space for one of these
  payloadSize -= sizeof(XtcData::Shape);  // Reserve space for the stale vector

  const size_t iNumPv = _lpvPvList.size();
  unsigned nWords = 1 + ((iNumPv - 1) >> 5);
  uint32_t* staleFlags = static_cast<uint32_t*>(desc.data());
  memset(staleFlags, 0, nWords * sizeof(*staleFlags));
  payloadSize -= nWords * sizeof(*staleFlags);

  std::vector<std::vector<uint32_t> > shapes(iNumPv);
  char* pXtc = reinterpret_cast<char*>(&staleFlags[nWords]);
  for (unsigned iPvName = 0; iPvName < iNumPv; iPvName++)
  {
    auto& epicsPvCur = *_lpvPvList[iPvName];

    if (_iDebugLevel >= 1)
      epicsPvCur.printPv();

    size_t size = payloadSize;
    bool stale;
    if (!epicsPvCur.addToXtc(stale, pXtc, size, shapes[iPvName]))
    {
      if (shapes[iPvName].size() != 0)         // If rank is non-zero,
        payloadSize -= sizeof(XtcData::Shape); // reserve space for Shape data

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

  // First, set data length...
  desc.set_data_length(pXtc - static_cast<char*>(desc.data()));

  // Second, set array shapes.  Can't do it in the other order
  uint32_t shape[XtcData::MaxRank];
  shape[0] = uint32_t(nWords);
  desc.set_array_shape(EpicsArchDef::Stale, shape);

  // Set array shape information for non-zero rank data
  for (unsigned iPvName = 0; iPvName < iNumPv; iPvName++)
  {
    const auto& epicsPvCur = *_lpvPvList[iPvName];
    if (!epicsPvCur.isDisabled() && (shapes[iPvName].size() != 0)) {
      desc.set_array_shape(EpicsArchDef::Data + iPvName, shapes[iPvName].data());
    }
  }

  return 0;     // All PV values are outputted successfully
}

unsigned EpicsArchMonitor::validate(unsigned& iPvCount, unsigned tmo)
{
  const size_t iNumPv = _lpvPvList.size();
  iPvCount = iNumPv;

  // Wait for PVs to connect
  std::chrono::seconds sTmo(tmo);
  auto t0(std::chrono::steady_clock::now());
  unsigned nNotConnected;
  do {
    nNotConnected = 0;
    for (unsigned iPvName = 0; iPvName < iNumPv; iPvName++)
    {
      EpicsMonitorPv& epicsPvCur = *_lpvPvList[iPvName];

      if (!epicsPvCur.isConnected()) {
        nNotConnected++;
      }
    }
    if (std::chrono::steady_clock::now() - t0 > sTmo)  break;
  } while (nNotConnected);

  // Check readiness and report on problematic PVs
  return validate(iPvCount);
}

unsigned EpicsArchMonitor::validate(unsigned& iPvCount)
{
  const size_t iNumPv = _lpvPvList.size();
  iPvCount = iNumPv;

  unsigned nNotConnected = 0;
  for (unsigned iPvName = 0; iPvName < iNumPv; iPvName++)
  {
    EpicsMonitorPv& epicsPvCur = *_lpvPvList[iPvName];

    if (epicsPvCur.isConnected()) {
      // Take select fields of interest and ignore uninteresting ones
      const std::string request("field(value,timeStamp,dimension)");

      unsigned tmo = 1;                 // Seconds
      if (!epicsPvCur.ready(request, tmo)) {
        epicsPvCur.disable();
        logging::warning("%s (%s) is not ready\n",
                         epicsPvCur.getPvDescription().c_str(), epicsPvCur.getPvName().c_str());
        nNotConnected++;
      }
    }
    else {
      epicsPvCur.disable();             //reconnect();
      logging::warning("%s (%s) is not connected\n",
                       epicsPvCur.getPvDescription().c_str(), epicsPvCur.getPvName().c_str());
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
    logging::warning("EpicsArchMonitor::_setupPvList(): Number of PVs (%zd) has reached capacity (%d), "
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
