#include <iostream>

#include "EpicsMonitorPv.hh"

#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "psalg/utils/SysLog.hh"

using logging = psalg::SysLog;


namespace Pds
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

  EpicsMonitorPv::EpicsMonitorPv(const std::string& sPvName,
                                 const std::string& sPvDescription,
                                 bool bProviderType) :
    Pds_Epics::MonTracker(sPvName),
    _sPvDescription(sPvDescription),
    _bProviderType(bProviderType),
    _bUpdated(false),
    _pData(nullptr),
    _size(0),
    _length(0)
  {
  }

  EpicsMonitorPv::~EpicsMonitorPv()
  {
    release();

    if (_pData)  free(_pData);
  }

  int EpicsMonitorPv::release()
  {
    if (isConnected())
      disconnect();

    return 0;
  }

  bool EpicsMonitorPv::ready(const std::string& request)
  {
    return getComplete(!_bProviderType ? CA : PVA, request);
  }

  int EpicsMonitorPv::addDef(EpicsArchDef& def, size_t& payloadSize)
  {
    const pvd::StructureConstPtr& structure = _strct->getStructure();
    if (!structure) {
      logging::critical("No payload for PV %s.  Is FieldMask empty?", name().c_str());
      std::string msg("No payload.  Is FieldMask empty?");
      throw msg;
    }
    const pvd::StringArray& names = structure->getFieldNames();
    const pvd::FieldConstPtrArray& fields = structure->getFields();
    unsigned i;
    for (i=0; i<fields.size(); i++) {
      if (names[i] == "value")  break;
    }
    std::string fullName(name() + "." + names[i]);
    switch (fields[i]->getType()) {
      case pvd::scalar: {
        const pvd::Scalar* scalar = static_cast<const pvd::Scalar*>(fields[i].get());
        XtcData::Name::DataType type = xtype[scalar->getScalarType()];
        def.NameVec.push_back(XtcData::Name(names[i].c_str(), type)); // Name must resolve to a name that psana recognizes: i.e. 'value'
        payloadSize = XtcData::Name::get_element_size(type);
        _pData = calloc(1, payloadSize);
        logging::info("PV name: %s  %s type: '%s' (%d)",
                      fullName.c_str(),
                      pvd::TypeFunc::name(fields[i]->getType()),
                      pvd::ScalarTypeFunc::name(scalar->getScalarType()),
                      type);
        switch (scalar->getScalarType()) {
          case pvd::pvInt:    getData = [&](void* data, size_t& length) -> size_t { return _getDatumT<int32_t >(data, length); };  break;
          case pvd::pvLong:   getData = [&](void* data, size_t& length) -> size_t { return _getDatumT<int64_t >(data, length); };  break;
          case pvd::pvUInt:   getData = [&](void* data, size_t& length) -> size_t { return _getDatumT<uint32_t>(data, length); };  break;
          case pvd::pvULong:  getData = [&](void* data, size_t& length) -> size_t { return _getDatumT<uint64_t>(data, length); };  break;
          case pvd::pvFloat:  getData = [&](void* data, size_t& length) -> size_t { return _getDatumT<float   >(data, length); };  break;
          case pvd::pvDouble: getData = [&](void* data, size_t& length) -> size_t { return _getDatumT<double  >(data, length); };  break;
          default: {
            logging::critical("%s: Unsupported %s type %s (%d)",
                              fullName.c_str(),
                              pvd::TypeFunc::name(fields[i]->getType()),
                              pvd::ScalarTypeFunc::name(scalar->getScalarType()),
                              scalar->getScalarType());
            throw "Unsupported scalar type";
            break;
          }
        }
        break;
      }

      case pvd::scalarArray: {
        const pvd::ScalarArray* array = static_cast<const pvd::ScalarArray*>(fields[i].get());
        XtcData::Name::DataType type = xtype[array->getElementType()];
        size_t nelem = _strct->getSubField<pvd::PVArray>(names[i].c_str())->getLength();
        def.NameVec.push_back(XtcData::Name(names[i].c_str(), type, 1)); // Name must resolve to a name that psana recognizes: i.e. 'value'
        payloadSize = nelem * XtcData::Name::get_element_size(type);
        _pData = calloc(1, payloadSize);
        logging::info("PV name: %s  %s type: %s (%d)  length: %zd",
                      fullName.c_str(),
                      pvd::TypeFunc::name(fields[i]->getType()),
                      pvd::ScalarTypeFunc::name(array->getElementType()),
                      type, nelem);
        switch (array->getElementType()) {
          case pvd::pvInt:    getData = [&](void* data, size_t& length) -> size_t { return _getDataT<int32_t >(data, length); };  break;
          case pvd::pvLong:   getData = [&](void* data, size_t& length) -> size_t { return _getDataT<int64_t >(data, length); };  break;
          case pvd::pvUInt:   getData = [&](void* data, size_t& length) -> size_t { return _getDataT<uint32_t>(data, length); };  break;
          case pvd::pvULong:  getData = [&](void* data, size_t& length) -> size_t { return _getDataT<uint64_t>(data, length); };  break;
          case pvd::pvFloat:  getData = [&](void* data, size_t& length) -> size_t { return _getDataT<float   >(data, length); };  break;
          case pvd::pvDouble: getData = [&](void* data, size_t& length) -> size_t { return _getDataT<double  >(data, length); };  break;
          default: {
            logging::critical("%s: Unsupported %s type %s (%d)",
                              fullName.c_str(),
                              pvd::TypeFunc::name(fields[i]->getType()),
                              pvd::ScalarTypeFunc::name(array->getElementType()),
                              array->getElementType());
            throw "Unsupported ScalarArray type";
            break;
          }
        }
        break;
      }

      default: {
        std::string msg("PV '"+name()+"' type '"+pvd::TypeFunc::name(fields[i]->getType())+
                        "' for field '"+names[i]+"' not supported");
        logging::warning("%s:  %s", __PRETTY_FUNCTION__, msg.c_str());
        //throw msg;
        break;
      }
    }

    std::string fnames("VarDef.NameVec fields: ");
    for (auto& elem: def.NameVec)
      fnames += std::string(elem.name()) + "[" + elem.str_type() + "],";
    logging::debug("%s",fnames.c_str());

    return 0;
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
      logging::error("EpicsMonitorPv::printPv(): Pv %s not Connected\n",
                     name().c_str());
      return 1;
    }

    printf("\n> PV %s\n", name().c_str());
    std::cout << "  channel:   " << _channel << "\n";
    std::cout << "  operation: " << _op      << "\n";
    std::cout << "  monitor:   " << _mon     << "\n";

    return 0;
  }

  int EpicsMonitorPv::addToXtc(bool& stale, char* pcXtcMem, size_t& iSizeXtc, size_t& iLength)
  {
    if (pcXtcMem == NULL)
      return 1;

    if (isConnected())
      _size = getData(_pData, _length);

    stale = !(isConnected() && _bUpdated);
    memcpy(pcXtcMem, _pData, _size);

    //printf("isConnected %d, updated %d, stale %c\n", isConnected(), _bUpdated, stale ? 'Y' : 'N');
    //printf("XtcSize %zd, length %zd\n", _size, _length);

    iSizeXtc = _size;
    iLength = _length;

    _bUpdated = false;

    return 0;
  }
}       // namespace Pds
