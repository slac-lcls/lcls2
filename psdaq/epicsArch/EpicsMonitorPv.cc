#include <iostream>

#include "EpicsMonitorPv.hh"

#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "psalg/utils/SysLog.hh"

using logging = psalg::SysLog;


namespace Drp
{

  static const XtcData::Name::DataType xtype[] = {
    XtcData::Name::UINT8 , // pvBoolean
    XtcData::Name::INT8  , // pvByte
    XtcData::Name::INT16 , // pvShort
    XtcData::Name::INT32 , // pvInt
    XtcData::Name::INT64 , // pvLong
    XtcData::Name::UINT8 , // pvUByte
    XtcData::Name::UINT16, // pvUShort
    XtcData::Name::UINT32, // pvUInt
    XtcData::Name::UINT64, // pvULong
    XtcData::Name::FLOAT , // pvFloat
    XtcData::Name::DOUBLE, // pvDouble
    XtcData::Name::CHARSTR, // pvString
  };

  // One mutex & condition variable shared by all EpicsMonitorPv instances
  std::mutex              EpicsMonitorPv::_mutex;
  std::condition_variable EpicsMonitorPv::_condition;

  EpicsMonitorPv::EpicsMonitorPv(const std::string& sPvName,
                                 const std::string& sPvDescription,
                                 const std::string& sProvider,
                                 const std::string& sRequest,
                                 bool               bDebug) :
    Pds_Epics::PvMonitorBase(sPvName, sProvider, sRequest),
    _sPvDescription(sPvDescription),
    _size(0),
    _pvField("value"),
    _state(NotReady),
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

  int EpicsMonitorPv::addVarDef(EpicsArchDef& varDef, size_t& size)
  {
    size = 0;

    if (_bDisabled)  return 1;

    {
      std::unique_lock<std::mutex> lock(_mutex);

      const std::chrono::seconds tmo(3);
      _condition.wait_for(lock, tmo, [this] { return _state == Ready; });
      if (_state != Ready)  return 1;

      auto& detName = !_sPvDescription.empty() ? _sPvDescription : _pvField;
      varDef.NameVec.push_back(XtcData::Name(detName.c_str(), xtype[_type], _rank));
      size = _pData.size();
    }

    if (_bDebug)
    {
      std::string fnames("VarDef.NameVec fields: ");
      for (auto& elem: varDef.NameVec)
        fnames += std::string(elem.name()) + "[" + elem.str_type() + "],";
      logging::debug("%s",fnames.c_str());
    }

    return 0;
  }

  void EpicsMonitorPv::onConnect()
  {
    logging::debug("%s connected\n", name().c_str());

    if (_bDebug)
      if (printStructure())
        logging::error("onConnect: printStructure() failed");
  }

  void EpicsMonitorPv::onDisconnect()
  {
    logging::warning("%s disconnected\n", name().c_str());
  }

  void EpicsMonitorPv::updated()
  {
    //logging::debug("EpicsMonitorPv::updated(): Called for '%s'", name().c_str());

    std::lock_guard<std::mutex> lock(_mutex);

    if (_state == Ready)
    {
      auto size = _pData.size();
      _shape = getData(_pData.data(), size);
      _size = size;
      _bUpdated = true;
    }
    else
    {
      if (getParams(_pvField, _type, _nelem, _rank) == 0)
      {
        _pData.resize(_nelem * XtcData::Name::get_element_size(xtype[_type]));

        _state = Ready;
      }
      else
        _bDisabled = true;

      _condition.notify_one();
    }
  }

  int EpicsMonitorPv::printPv() const
  {
    if (!isConnected())
    {
      logging::error("EpicsMonitorPv::printPv(): PV %s is not Connected",
                     name().c_str());
      return 1;
    }

    std::cout << "\n> PV "       << name()   << "\n";
    std::cout << "  channel:   " << _channel << "\n";
    std::cout << "  monitor:   " << _mon     << "\n";

    return 0;
  }

  int EpicsMonitorPv::addToXtc(XtcData::Damage& damage, bool& stale, char* pcXtcMem, size_t& iSizeXtc, std::vector<uint32_t>& sShape)
  {
    stale = _bDisabled;

    if (_bDisabled)
      return 1;

    std::lock_guard<std::mutex> lock(_mutex);

    auto sizeXtc = _pData.size();
    if (sizeXtc > iSizeXtc)  sizeXtc = iSizeXtc; // Possibly truncate
    memcpy(pcXtcMem, _pData.data(), sizeXtc);

    if (_size > _pData.size())
    {
      logging::error("EpicsMonitorPv::updated: %s data truncated; size %zu vs %zu\n",
                     name().c_str(), _pData.size(), _size);
      damage.increase(XtcData::Damage::Truncated);
      _pData.resize(_size);
    }

    // Consider the PV stale if it hasn't updated since the last addToXtc call
    stale = !_bUpdated;
    if (stale)  logging::debug("PV is stale: %s\n", name().c_str());

    //printf("XtcSize %zd, shape ", sizeXtc);
    //for (auto dim: _shape)  printf("%d ", dim);
    //printf("\n");

    iSizeXtc = sizeXtc;
    sShape   = _shape;

    _bUpdated = false;

    return 0;
  }
}       // namespace Drp
