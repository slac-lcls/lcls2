#include "PvMonitorBase.hh"

#include "psalg/utils/SysLog.hh"

#include "pv/pvTimeStamp.h"
#include "pv/pvEnumerated.h"

using logging = psalg::SysLog;

namespace Pds_Epics
{

typedef uint32_t ShpArr[PvMonitorBase::MaxRank];

int PvMonitorBase::printStructure() const
{
    if (!_strct)  { logging::error("_strct is NULL");  return 1; }
    const auto& structure = _strct->getStructure();
    if (!structure)  { logging::error("structure is NULL");  return 1; }
    const auto& names = structure->getFieldNames();
    //std::cout << "Fields are: " << structure << "\n";
    for (unsigned i=0; i<names.size(); i++) {
        const auto& pvField = _strct->getSubField<pvd::PVField>(names[i]);
        if (!pvField)  { logging::error("pvField %s is NULL", names[i].c_str());  return 1; }
        const auto& field  = pvField->getField();
        const auto& offset = pvField->getFieldOffset();
        logging::info("PV Name: %s  FieldName: %s  Offset: %zu  FieldType: %s  ID: %s",
                      name().c_str(), names[i].c_str(), offset,
                      pvd::TypeFunc::name(field->getType()),
                      field->getID().c_str());
        switch (field->getType()) {
            case pvd::scalar: {
                const auto& pvScalar = _strct->getSubField<pvd::PVScalar>(offset);
                if (!pvScalar)  { logging::error("pvScalar is NULL");  return 1; }
                //std::cout << "PVScalar: " << pvScalar << "\n";
                const auto& scalar   = pvScalar->getScalar();
                if (!scalar)  { logging::error("scalar is NULL");  return 1; }
                //std::cout << "Scalar: " << scalar << "\n";
                const auto& fType = pvd::ScalarTypeFunc::name(scalar->getScalarType());
                std::cout << "  Scalar type: " << fType << "\n";
                break;
            }
            case pvd::scalarArray: {
                const auto& pvScalarArray = _strct->getSubField<pvd::PVScalarArray>(offset);
                if (!pvScalarArray)  { logging::error("pvScalarArray is NULL");  return 1; }
                //std::cout << "PVScalarArray: " << pvScalarArray << "\n";
                const auto& scalarArray   = pvScalarArray->getScalarArray();
                if (!scalarArray)  { logging::error("scalarArray is NULL");  return 1; }
                //std::cout << "ScalarArray: " << scalarArray << "\n";
                const auto& fType         = pvd::ScalarTypeFunc::name(scalarArray->getElementType());
                std::cout << "  ScalarArray type: " << fType << "\n";
                break;
            }
            case pvd::structure: {
                const auto& pvStructure = _strct->getSubField<pvd::PVStructure>(offset);
                if (!pvStructure)  { logging::error("pvStructure is NULL");  return 1; }
                //std::cout << "PVStructure: " << pvStructure << "\n";
                const auto& structure   = pvStructure->getStructure();
                if (!structure)  { logging::error("structure is NULL");  return 1; }
                //std::cout << "  Structure: " << structure << "\n";
                const auto& fieldNames  = structure->getFieldNames();
                for (unsigned k = 0; k < fieldNames.size(); k++) {
                    const auto& pvField = pvStructure->getSubField(fieldNames[k]);
                    const auto& offset  = pvField->getFieldOffset();
                    const auto& field   = pvField->getField();
                    const auto& fType   = field->getType();
                    std::cout << "    field '" << fieldNames[k] << "', offset " << offset
                              << ", type " << fType << ", subtype " << field << "\n";
                }
                break;
            }
            case pvd::structureArray: {
                const auto& pvStructureArray = _strct->getSubField<pvd::PVStructureArray>(offset);
                if (!pvStructureArray)  { logging::error("pvStructureArray is NULL");  return 1; }
                //std::cout << "PVStructureArray: " << pvStructureArray << "\n";
                const auto& structureArray   = pvStructureArray->getStructureArray();
                if (!structureArray)  { logging::error("structureArray is NULL");  return 1; }
                //std::cout << "StructureArray: " << structureArray << "\n";
                printf("  StructureArray has length %zu\n", pvStructureArray->getLength());
                std::vector<int> sizes(pvStructureArray->getLength());
                const auto& pvStructure      = pvStructureArray->view();
                for (unsigned j = 0; j < pvStructureArray->getLength(); ++j) {
                    std::cout << "  PVStructure: " << pvStructure[j] << "\n";
                    const auto& fieldNames = pvStructure[j]->getStructure()->getFieldNames();
                    for (unsigned k = 0; k < fieldNames.size(); k++) {
                        const auto& pvField = pvStructure[j]->getSubField(fieldNames[k]);
                        if (!pvField)  { logging::error("pvField is NULL");  return 1; }
                        if (names[i] == "dimension" && fieldNames[k] == "size") {
                            const auto& pvInt = pvStructure[j]->getSubField<pvd::PVInt>("size");
                            sizes[j] = pvInt ? pvInt->getAs<int>() : 0;
                        }
                        else {
                            printf("    Non-'%s' field '%s', offset %zu\n", "size", fieldNames[k].c_str(), pvField->getFieldOffset());
                        }
                    }
                }
                if (names[i] == "dimension") {
                    for (unsigned j = 0; j < sizes.size(); ++j) {
                        printf("  PVStructure[%u] size: %d\n", j, sizes[j]);
                    }
                }
                break;
            }
            case pvd::union_: {
                const auto& pvUnion = _strct->getSubField<pvd::PVUnion>(offset);
                if (!pvUnion)  { logging::error("pvUnion is NULL");  return 1; }
                //std::cout << "PVUnion: " << pvUnion << "\n";
                const auto& union_  = pvUnion->getUnion();
                if (!union_)  { logging::error("union is NULL");  return 1; }
                //std::cout << "  Union: " << union_ << "\n";
                printf("  Union has %zu fields\n", union_->getNumberFields());
                printf("  Union is variant: %d\n", union_->isVariant());
                std::cout << "  PVUnion numberFields: " << pvUnion->getNumberFields() << "\n";
                const auto& pvField = pvUnion->get();
                if (!pvField)  { logging::error("pvField is NULL");  return 1; }
                const auto& field   = pvField->getField();
                std::cout << "  PVUnion type: " << field->getType() << "\n";
                std::cout << "  PVUnion subtype: " << field->getID() << "\n";
                const auto& pvScalarArray = pvUnion->get<pvd::PVScalarArray>();
                if (!pvScalarArray)  { logging::error("Union's pvScalarArray is NULL");  return 1; }
                //std::cout << "PVScalarArray: " << pvScalarArray << "\n";
                const auto& scalarArray   = pvScalarArray->getScalarArray();
                if (!scalarArray)  { logging::error("Union's scalarArray is NULL");  return 1; }
                //std::cout << "ScalarArray: " << scalarArray << "\n";
                const auto& fType         = pvd::ScalarTypeFunc::name(scalarArray->getElementType());
                std::cout << "  ScalarArray offset: " << pvScalarArray->getFieldOffset() << "  Type: " << fType << "\n";
                break;
            }
            case pvd::unionArray: {     // Placeholder for now
                const auto& pvUnionArray = _strct->getSubField<pvd::PVUnion>(offset);
                if (!pvUnionArray)  { logging::error("pvUnionArray is NULL");  return 1; }
                std::cout << "PVUnionArray: " << pvUnionArray << "\n";
                const auto& unionArray   = pvUnionArray->getUnion();
                if (!unionArray)  { logging::error("unionArray is NULL");  return 1; }
                std::cout << "  UnionArray: " << unionArray << "\n";
                logging::error("UnionArrays are unsupported as yet");
                break;
            }
            default: {
                break;
            }
        }
    }

    return 0;
}

int PvMonitorBase::getParams(pvd::ScalarType& type,
                             size_t&          nelem,
                             size_t&          rank)
{
    if (!_strct)  { logging::error("_strct is NULL");  return 1; }
    const auto& pvStructureArray = _strct->getSubField<pvd::PVStructureArray>("dimension");
    rank = pvStructureArray ? pvStructureArray->getLength() : 1;

    const auto& pvField = _strct->getSubField<pvd::PVField>(m_fieldName);
    if (!pvField) {
      logging::critical("No payload for PV %s.  Is FieldMask empty?", MonTracker::name().c_str());
      throw "No payload.  Is FieldMask empty?";
    }
    const auto& field   = pvField->getField();
    const auto& offset  = pvField->getFieldOffset();
    switch (field->getType()) {
        case pvd::scalar: {
            const auto& pvScalar = _strct->getSubField<pvd::PVScalar>(offset);
            if        (!pvScalar)  { logging::error("pvScalar is NULL");  return 1; }
            const auto& scalar   = pvScalar->getScalar();
            if        (!scalar)    { logging::error("scalar is NULL");    return 1; }
            type  = scalar->getScalarType();
            nelem = type != pvd::pvString ? 1 : MAX_STRING_SIZE;
            rank  = type != pvd::pvString ? 0 : 1;
            // This was tested and is known to work
            switch (type) {
                case pvd::pvBoolean: getData = [&](void* buf, size_t len, ShpArr shp) -> size_t { return _getScalar<uint8_t >(buf, len, shp); };  break;
                case pvd::pvByte:    getData = [&](void* buf, size_t len, ShpArr shp) -> size_t { return _getScalar<int8_t  >(buf, len, shp); };  break;
                case pvd::pvShort:   getData = [&](void* buf, size_t len, ShpArr shp) -> size_t { return _getScalar<int16_t >(buf, len, shp); };  break;
                case pvd::pvInt:     getData = [&](void* buf, size_t len, ShpArr shp) -> size_t { return _getScalar<int32_t >(buf, len, shp); };  break;
                case pvd::pvLong:    getData = [&](void* buf, size_t len, ShpArr shp) -> size_t { return _getScalar<int64_t >(buf, len, shp); };  break;
                case pvd::pvUByte:   getData = [&](void* buf, size_t len, ShpArr shp) -> size_t { return _getScalar<uint8_t >(buf, len, shp); };  break;
                case pvd::pvUShort:  getData = [&](void* buf, size_t len, ShpArr shp) -> size_t { return _getScalar<uint16_t>(buf, len, shp); };  break;
                case pvd::pvUInt:    getData = [&](void* buf, size_t len, ShpArr shp) -> size_t { return _getScalar<uint32_t>(buf, len, shp); };  break;
                case pvd::pvULong:   getData = [&](void* buf, size_t len, ShpArr shp) -> size_t { return _getScalar<uint64_t>(buf, len, shp); };  break;
                case pvd::pvFloat:   getData = [&](void* buf, size_t len, ShpArr shp) -> size_t { return _getScalar<float   >(buf, len, shp); };  break;
                case pvd::pvDouble:  getData = [&](void* buf, size_t len, ShpArr shp) -> size_t { return _getScalar<double  >(buf, len, shp); };  break;
                case pvd::pvString:  getData = [&](void* buf, size_t len, ShpArr shp) -> size_t { return _getScalarString    (buf, len, shp); };  break;
                default: {
                    logging::critical("%s: Unsupported %s type %s (%d)",
                                      MonTracker::name().c_str(),
                                      pvd::TypeFunc::name(field->getType()),
                                      pvd::ScalarTypeFunc::name(type),
                                      type);
                    throw "Unsupported Scalar type";
                }
            }
            logging::info("PV name: %s,  %s type: '%s' (%d),  length: %zd,  rank: %zd",
                          MonTracker::name().c_str(),
                          pvd::TypeFunc::name(field->getType()),
                          pvd::ScalarTypeFunc::name(type),
                          type, nelem, rank);
            break;
        }
        case pvd::scalarArray: {
            const auto& pvScalarArray = _strct->getSubField<pvd::PVScalarArray>(offset);
            if        (!pvScalarArray)  { logging::error("pvScalarArray is NULL");  return 1; }
            const auto& scalarArray   = pvScalarArray->getScalarArray();
            if        (!scalarArray)    { logging::error("scalarArray is NULL");    return 1; }
            type  = scalarArray->getElementType();
            nelem = (type != pvd::pvString ?    1 : MAX_STRING_SIZE) * pvScalarArray->getLength();
            rank  =  type != pvd::pvString ? rank : 2;
            // This was tested and is known to work
            switch (type) {
                case pvd::pvBoolean: getData = [&](void* buf, size_t len, ShpArr shp) -> size_t { return _getArray<uint8_t >(buf, len, shp); };  break;
                case pvd::pvByte:    getData = [&](void* buf, size_t len, ShpArr shp) -> size_t { return _getArray<int8_t  >(buf, len, shp); };  break;
                case pvd::pvShort:   getData = [&](void* buf, size_t len, ShpArr shp) -> size_t { return _getArray<int16_t >(buf, len, shp); };  break;
                case pvd::pvInt:     getData = [&](void* buf, size_t len, ShpArr shp) -> size_t { return _getArray<int32_t >(buf, len, shp); };  break;
                case pvd::pvLong:    getData = [&](void* buf, size_t len, ShpArr shp) -> size_t { return _getArray<int64_t >(buf, len, shp); };  break;
                case pvd::pvUByte:   getData = [&](void* buf, size_t len, ShpArr shp) -> size_t { return _getArray<uint8_t >(buf, len, shp); };  break;
                case pvd::pvUShort:  getData = [&](void* buf, size_t len, ShpArr shp) -> size_t { return _getArray<uint16_t>(buf, len, shp); };  break;
                case pvd::pvUInt:    getData = [&](void* buf, size_t len, ShpArr shp) -> size_t { return _getArray<uint32_t>(buf, len, shp); };  break;
                case pvd::pvULong:   getData = [&](void* buf, size_t len, ShpArr shp) -> size_t { return _getArray<uint64_t>(buf, len, shp); };  break;
                case pvd::pvFloat:   getData = [&](void* buf, size_t len, ShpArr shp) -> size_t { return _getArray<float   >(buf, len, shp); };  break;
                case pvd::pvDouble:  getData = [&](void* buf, size_t len, ShpArr shp) -> size_t { return _getArray<double  >(buf, len, shp); };  break;
                case pvd::pvString:  getData = [&](void* buf, size_t len, ShpArr shp) -> size_t { return _getArrayString    (buf, len, shp); };  break;
                default: {
                    logging::critical("%s: Unsupported %s type '%s' (%d)",
                                      MonTracker::name().c_str(),
                                      pvd::TypeFunc::name(field->getType()),
                                      pvd::ScalarTypeFunc::name(type),
                                      type);
                    throw "Unsupported ScalarArray type";
                }
            }
            logging::info("PV name: %s,  %s type: '%s' (%d),  length: %zd,  rank: %zd",
                          MonTracker::name().c_str(),
                          pvd::TypeFunc::name(field->getType()),
                          pvd::ScalarTypeFunc::name(type),
                          type, nelem, rank);
            break;
        }
        case pvd::structure: {
            if (field->getID() == "enum_t") {
                const auto& pvStructure = _strct->getSubField<pvd::PVStructure>(offset);
                if        (!pvStructure)  { logging::error("pvStructure is NULL");  return 1; }
                const auto& structure   = pvStructure->getStructure();
                if        (!structure)    { logging::error("structure is NULL");    return 1; }
                type  = pvd::pvString;
                rank  = 1;
                nelem = MAX_STRING_SIZE;
                getData = [&](void* buf, size_t len, ShpArr shp) -> size_t { return _getEnum(buf, len, shp); };
                logging::info("PV name: %s,  %s type: '%s' (%d),  length: %zd,  rank: %zd",
                              MonTracker::name().c_str(),
                              pvd::TypeFunc::name(field->getType()),
                              pvd::ScalarTypeFunc::name(type),
                              type, nelem, rank);
            }
            else
            {
                logging::critical("%s: Unsupported Structure ID '%s'",
                                  MonTracker::name().c_str(), field->getID().c_str());
                throw "Unsupported Structure ID";
            }
            break;
        }
        // case pvd::structureArray: {  // Placeholder
        //     break;
        // }
        case pvd::union_: {
            const auto& pvUnion       = _strct->getSubField<pvd::PVUnion>(offset);
            if        (!pvUnion)        { logging::error("pvUnion is NULL");                return 1; }
            //const auto& union_        = pvUnion->getUnion();
            //if        (!union_)         { logging::error("union is NULL");                  return 1; }
            const auto& pvField       = pvUnion->get();
            if        (!pvField)        { logging::error("pvField is NULL");                return 1; }
            const auto& ufield        = pvField->getField();
            if        (!ufield)         { logging::error("ufield is NULL");                 return 1; }
            switch (ufield->getType()) {
                //case pvd::scalar: {     // Place holder
                //    break;
                //}
                case pvd::scalarArray: {
                    const auto& pvScalarArray = pvUnion->get<pvd::PVScalarArray>();
                    if        (!pvScalarArray)  { logging::error("Union's pvScalarArray is NULL");  return 1; }
                    const auto& scalarArray   = pvScalarArray->getScalarArray();
                    if        (!scalarArray)    { logging::error("Union's scalarArray is NULL");    return 1; }
                    if (pvUnion->getNumberFields() != 1) {
                        logging::critical("%s: Unsupported Union ScalarArray field count %d\n",
                                          MonTracker::name().c_str(), pvUnion->getNumberFields());
                        throw "Unsupported Union ScalarArray field count";
                    }
                    type  = scalarArray->getElementType();
                    nelem = pvScalarArray->getLength();
                    // This has NOT been tested
                    switch (type) {
                        case pvd::pvBoolean: getData = [&](void* buf, size_t len, ShpArr shp) -> size_t { return _getUnionSclArr<uint8_t >(buf, len, shp); };  break;
                        case pvd::pvByte:    getData = [&](void* buf, size_t len, ShpArr shp) -> size_t { return _getUnionSclArr<int8_t  >(buf, len, shp); };  break;
                        case pvd::pvShort:   getData = [&](void* buf, size_t len, ShpArr shp) -> size_t { return _getUnionSclArr<int16_t >(buf, len, shp); };  break;
                        case pvd::pvInt:     getData = [&](void* buf, size_t len, ShpArr shp) -> size_t { return _getUnionSclArr<int32_t >(buf, len, shp); };  break;
                        case pvd::pvLong:    getData = [&](void* buf, size_t len, ShpArr shp) -> size_t { return _getUnionSclArr<int64_t >(buf, len, shp); };  break;
                        case pvd::pvUByte:   getData = [&](void* buf, size_t len, ShpArr shp) -> size_t { return _getUnionSclArr<uint8_t >(buf, len, shp); };  break;
                        case pvd::pvUShort:  getData = [&](void* buf, size_t len, ShpArr shp) -> size_t { return _getUnionSclArr<uint16_t>(buf, len, shp); };  break;
                        case pvd::pvUInt:    getData = [&](void* buf, size_t len, ShpArr shp) -> size_t { return _getUnionSclArr<uint32_t>(buf, len, shp); };  break;
                        case pvd::pvULong:   getData = [&](void* buf, size_t len, ShpArr shp) -> size_t { return _getUnionSclArr<uint64_t>(buf, len, shp); };  break;
                        case pvd::pvFloat:   getData = [&](void* buf, size_t len, ShpArr shp) -> size_t { return _getUnionSclArr<float   >(buf, len, shp); };  break;
                        case pvd::pvDouble:  getData = [&](void* buf, size_t len, ShpArr shp) -> size_t { return _getUnionSclArr<double  >(buf, len, shp); };  break;
                        default:{
                            logging::critical("%s: Unsupported %s %s type '%s' (%d)",
                                              MonTracker::name().c_str(),
                                              pvd::TypeFunc::name(field->getType()),
                                              pvd::TypeFunc::name(ufield->getType()),
                                              pvd::ScalarTypeFunc::name(type),
                                              type);
                            throw "Unsupported Union ScalarArray type";
                        }
                    }
                    break;
                }
                default: {
                    logging::critical("%s: Unsupported %s type '%s' for subfield '%s'",
                                      MonTracker::name().c_str(),
                                      pvd::TypeFunc::name(field->getType()),
                                      pvd::TypeFunc::name(ufield->getType()),
                                      pvField->getFieldName().c_str());
                    throw "Unsupported Union type";
                }
            }
            logging::info("PV name: %s,  %s type: '%s' (%d),  length: %zd,  rank: %zd",
                          MonTracker::name().c_str(),
                          pvd::TypeFunc::name(field->getType()),
                          pvd::ScalarTypeFunc::name(type),
                          type, nelem, rank);
            break;
        }
        //case pvd::unionArray: {         // Place holder
        //    break;
        //}
        default: {
            logging::critical("%s: Unsupported field type '%s' for subfield '%s'",
                              MonTracker::name().c_str(),
                              pvd::TypeFunc::name(field->getType()),
                              pvField->getFieldName().c_str());
            throw "Unsupported field type";
        }
    }

    return 0;
}

void PvMonitorBase::_getDimensions(uint32_t shape[MaxRank]) const
{
    const auto& pvStructureArray = _strct->getSubField<pvd::PVStructureArray>("dimension");
    if (pvStructureArray) {
        const auto& pvStructure = pvStructureArray->view();
        const auto  ranks       = pvStructureArray->getLength();
        if (ranks > MaxRank) {
          logging::critical("%s: Unsupported number of dimensions %zs\n",
                            MonTracker::name().c_str(), ranks);
          throw "Unsupported number of shape dimensions";
        }
        for (unsigned i = 0; i < ranks; ++i) {
            // EPICS data shows up in [x,y] order but psana wants [y,x]
            shape[i] = pvStructure[ranks - 1 - i]->getSubField<pvd::PVInt>("size")->getAs<uint32_t>();
        }
    }
    // else leave the already calculated shape alone
}

template<typename T>
size_t PvMonitorBase::_getScalar(std::shared_ptr<const pvd::PVScalar> const& pvScalar, void* data, size_t size, uint32_t shape[MaxRank]) const {
    if (sizeof(T) <= size)
      *static_cast<T*>(data) = _strct->getSubField<pvd::PVScalar>(m_fieldName)->getAs<T>();
    return sizeof(T);
}

static size_t copyString(void* dst, const std::string& src, size_t size)
{
    auto buf = static_cast<char*>(dst);
    strncpy(buf, src.c_str(), size);    // Possibly truncates
    auto sz  = src.size() + 1;          // +1 for null terminator
    if  (sz >= size)  buf[size - 1] = '\0';
    return sz;
}

size_t PvMonitorBase::_getScalarString(void* data, size_t size, uint32_t shape[MaxRank]) const
{
    const auto& sval(_strct->getSubField<pvd::PVScalar>(m_fieldName)->getAs<std::string>());
    auto sz = copyString(data, sval, size);
    shape[0] = sz < size ? sz : size;   // Rank 1 array, includes null
    return sz;                          // Bytes needed to avoid truncation
}

template<typename T>
size_t PvMonitorBase::_getArray(std::shared_ptr<const pvd::PVScalarArray> const& pvScalarArray, void* data, size_t size, uint32_t shape[MaxRank]) const {
    //pvd::shared_vector<const T> vec((T*)data, [](void*){}, 0, 128); // Doesn't work: tries to delete [] data
    pvd::shared_vector<const T> vec;
    pvScalarArray->getAs<T>(vec);
    auto count = vec.size();
    auto sz    = count * sizeof(T);
    memcpy(data, vec.data(), sz < size ? sz : size); // Possibly truncates
    shape[0] = count;                   // Number of elements in rank 1 array
    return sz;                          // Bytes needed to avoid truncation
}

size_t PvMonitorBase::_getArrayString(void* data, size_t size, uint32_t shape[MaxRank]) const
{
    pvd::shared_vector<const std::string> vec;
    _strct->getSubField<pvd::PVScalarArray>(m_fieldName)->getAs<std::string>(vec);
    auto   count = vec.size();
    size_t szPer = size / count;
    char*  buf   = static_cast<char*>(data);
    size_t maxSz = 0;
    for (unsigned i = 0; i < count; ++i) {
        auto sz = copyString(buf, vec[i], szPer);
        if (sz > maxSz)   maxSz = sz;
        buf += szPer;
    }
    shape[0] = count;                   // Number of    elements in rank 2 array
    shape[1] = szPer;                   // Size of each element  in rank 2 array
    return count * maxSz;               // Bytes needed to avoid truncation
}

size_t PvMonitorBase::_getEnum(void* data, size_t size, uint32_t shape[MaxRank]) const {
    pvd::shared_vector<const std::string> choices;
    const auto& pvStructure = _strct->getSubField<pvd::PVStructure>(m_fieldName);
    auto idx = pvStructure->getSubField<pvd::PVScalar>("index")->getAs<int>();
    pvStructure->getSubField<pvd::PVScalarArray>("choices")->getAs<std::string>(choices);

    auto sz = copyString(data, choices[idx], size);
    shape[0] = sz < size ? sz : size;   // Rank 1 array, includes null
    return sz;                          // Bytes needed to avoid truncation
}

} // namespace Pds_Epics
