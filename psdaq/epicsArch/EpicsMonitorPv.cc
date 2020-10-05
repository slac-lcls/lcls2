#include <iostream>

#include "EpicsMonitorPv.hh"

#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "psalg/utils/SysLog.hh"

using logging = psalg::SysLog;


namespace Drp
{

  EpicsMonitorPv::EpicsMonitorPv(const std::string& sPvName,
                                 const std::string& sPvDescription,
                                 const std::string& sProvider,
                                 bool               bDebug) :
    Drp::PvMonitorBase(sPvName, sProvider),
    _sPvDescription(sPvDescription),
    _size(0),
    _bUpdated(false),
    _bDisabled(false),
    _bDebug(bDebug)
  {
  }

  EpicsMonitorPv::~EpicsMonitorPv()
  {
    release();
  }

  int EpicsMonitorPv::release()
  {
    if (isConnected())
      disconnect();

    return 0;
  }

  int EpicsMonitorPv::addDef(EpicsArchDef& def)
  {
    if (_bDisabled)  return 1;

    std::string             name = "value";
    XtcData::Name::DataType type;
    size_t                  size;
    size_t                  rank;
    getParams(name, type, size, rank);

    auto detName = !_sPvDescription.empty() ? _sPvDescription : name;

    def.NameVec.push_back(XtcData::Name(detName.c_str(), type, rank));
    _pData.resize(size);

    std::string fnames("VarDef.NameVec fields: ");
    for (auto& elem: def.NameVec)
      fnames += std::string(elem.name()) + "[" + elem.str_type() + "],";
    logging::debug("%s",fnames.c_str());

    return 0;
  }

  void EpicsMonitorPv::onConnect()
  {
    logging::info("%s connected\n", name().c_str());

    if (_bDebug)
      printStructure();
  }

  void EpicsMonitorPv::onDisconnect()
  {
    logging::info("%s disconnected\n", name().c_str());
  }

  void EpicsMonitorPv::updated()
  {
    //logging::debug("EpicsMonitorPv::updated(): Called for '%s'", name().c_str());

    _bUpdated = true;
  }

  int EpicsMonitorPv::printPv() const
  {
    if (!isConnected())
    {
      logging::error("EpicsMonitorPv::printPv(): PV %s not Connected",
                     name().c_str());
      return 1;
    }

    printf("\n> PV %s\n", name().c_str());
    std::cout << "  channel:   " << _channel << "\n";
    std::cout << "  operation: " << _op      << "\n";
    std::cout << "  monitor:   " << _mon     << "\n";

    return 0;
  }

  int EpicsMonitorPv::addToXtc(bool& stale, char* pcXtcMem, size_t& iSizeXtc, std::vector<unsigned>& sShape)
  {
    if (pcXtcMem == NULL || _bDisabled)
      return 1;

    auto size = _pData.size();
    if (isConnected())
    {
      _shape = getData(_pData.data(), size);
      _size = size;
    }

    auto sizeXtc = _size;
    if (sizeXtc > iSizeXtc)  sizeXtc = iSizeXtc; // Possibly truncate
    memcpy(pcXtcMem, _pData.data(), sizeXtc);

    if (size > _pData.size())  _pData.resize(size);

    stale = !(isConnected() && _bUpdated && !_bDisabled);

    //printf("isConnected %d, updated %d, stale %c\n", isConnected(), _bUpdated, stale ? 'Y' : 'N');
    //printf("XtcSize %zd, shape ", _size);
    //for (auto dim: _shape)  printf("%d ", dim);
    //printf("\n");

    iSizeXtc = _size;
    sShape   = _shape;

    _bUpdated = false;

    return 0;
  }
}       // namespace Drp
