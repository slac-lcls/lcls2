#include "PvMonitorBase.hh"

#include "psalg/utils/SysLog.hh"

#include "pv/pvTimeStamp.h"

using logging = psalg::SysLog;

namespace Pds_Epics
{

void PvMonitorBase::printStructure() const
{
    auto structure = _strct->getStructure();
    auto names = structure->getFieldNames();
    auto fields = structure->getFields();
    //std::cout << "Fields are: " << structure << "\n";
    for (unsigned i=0; i<names.size(); i++) {
        auto field = _strct->getSubField<pvd::PVField>(names[i]);
        auto offset = field->getFieldOffset();
        logging::info("PV Name: %s  FieldName: %s  Offset: %zu  FieldType: %s",
                      name().c_str(), names[i].c_str(), offset, pvd::TypeFunc::name(fields[i]->getType()));
        std::cout << "Field type: " << field->getField()->getType() << "\n";
        switch (fields[i]->getType()) {
            case pvd::scalar: {
                auto pvScalar = _strct->getSubField<pvd::PVScalar>(offset);
                //std::cout << "PVScalar: " << pvScalar << "\n";
                auto scalar   = pvScalar->getScalar();
                auto fType = pvd::ScalarTypeFunc::name(scalar->getScalarType());
                std::cout << "  Scalar type: " << fType << "\n";
                break;
            }
            case pvd::scalarArray: {
                auto pvScalarArray = _strct->getSubField<pvd::PVScalarArray>(offset);
                //std::cout << "PVScalarArray: " << pvScalarArray << "\n";
                auto scalarArray   = pvScalarArray->getScalarArray();
                auto fType = pvd::ScalarTypeFunc::name(scalarArray->getElementType());
                std::cout << "  ScalarArray type: " << fType << "\n";
                break;
            }
            case pvd::union_: {
                auto pvUnion = _strct->getSubField<pvd::PVUnion>(offset);
                //std::cout << "PVUnion: " << pvUnion << "\n";
                auto union_   = pvUnion->getUnion();
                //std::cout << "  Union: " << union_ << "\n";
                printf("  Union has %zu fields\n", union_->getNumberFields());
                printf("  Union is variant: %d\n", union_->isVariant());
                std::cout << "  PVUnion numberFields: " << pvUnion->getNumberFields() << "\n";
                std::cout << "  PVUnion type: " << pvUnion->get()->getField()->getType() << "\n";
                std::cout << "  PVUnion subtype: " << pvUnion->get()->getField()->getID() << "\n";
                auto pvScalarArray = pvUnion->get<pvd::PVScalarArray>();
                auto scalarArray   = pvScalarArray->getScalarArray();
                auto fType = pvd::ScalarTypeFunc::name(scalarArray->getElementType());
                std::cout << "  ScalarArray offset: " << pvScalarArray->getFieldOffset() << "  Type: " << fType << "\n";
                break;
            }
            case pvd::structure: {
                auto pvStructure = _strct->getSubField<pvd::PVStructure>(offset);
                //std::cout << "PVStructure: " << pvStructure << "\n";
                auto structure   = pvStructure->getStructure();
                //std::cout << "  Structure: " << structure << "\n";
                auto fieldNames = structure->getFieldNames();
                for (unsigned k = 0; k < fieldNames.size(); k++) {
                    auto pvField = pvStructure->getSubField(fieldNames[k]);
                    auto offset = pvField->getFieldOffset();
                    printf("    field '%s' has offset %zu\n", fieldNames[k].c_str(), offset);
                }
                break;
            }
            case pvd::structureArray: {
                auto pvStructureArray = _strct->getSubField<pvd::PVStructureArray>(offset);
                std::cout << "PVStructureArray: " << pvStructureArray << "\n";
                auto structureArray   = pvStructureArray->getStructureArray();
                //std::cout << "StructureArray: " << structureArray << "\n";
                printf("  StructureArray has length %zu\n", pvStructureArray->getLength());
                std::vector<int> sizes(pvStructureArray->getLength());
                auto pvStructure = pvStructureArray->view();
                for (unsigned j = 0; j < pvStructureArray->getLength(); ++j) {
                    std::cout << "  PVStructure: " << pvStructure[j] << "\n";
                    auto fieldNames = pvStructure[j]->getStructure()->getFieldNames();
                    for (unsigned k = 0; k < fieldNames.size(); k++) {
                        auto pvField = pvStructure[j]->getSubField(fieldNames[k]);
                        if (names[i] == "dimension" && fieldNames[k] == "size") {
                            auto pvInt = pvStructure[j]->getSubField<pvd::PVInt>("size");
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
            default: {
                break;
            }
        }
    }
}

int PvMonitorBase::getParams(const std::string& name,
                             pvd::ScalarType&   type,
                             size_t&            nelem,
                             size_t&            rank)
{
    auto pvStructureArray = _strct->getSubField<pvd::PVStructureArray>("dimension");
    rank = pvStructureArray ? pvStructureArray->getLength() : 1;

    auto field  = _strct->getSubField<pvd::PVField>(name);
    if (!field) {
      logging::critical("No payload for PV %s.  Is FieldMask empty?", MonTracker::name().c_str());
      throw "No payload.  Is FieldMask empty?";
    }
    auto offset = field->getFieldOffset();
    auto fullName(MonTracker::name() + "." + field->getFullName());
    switch (field->getField()->getType()) {
        case pvd::scalar: {
            auto pvScalar = _strct->getSubField<pvd::PVScalar>(offset);
            if (!pvScalar)  throw "pvScalar is NULL";
            auto scalar   = pvScalar->getScalar();
            if (!scalar)  throw "scalar is NULL";
            type          = scalar->getScalarType();
            nelem         = 1;
            rank          = 0;
// This was tested and is known to work
//          switch (scalar->getScalarType()) {
//              case pvd::pvBoolean: getData = [&](void* data, size_t& length) -> size_t { return _getScalar<uint8_t >(data, length); };  break;
//              case pvd::pvByte:    getData = [&](void* data, size_t& length) -> size_t { return _getScalar<int8_t  >(data, length); };  break;
//              case pvd::pvShort:   getData = [&](void* data, size_t& length) -> size_t { return _getScalar<int16_t >(data, length); };  break;
//              case pvd::pvInt:     getData = [&](void* data, size_t& length) -> size_t { return _getScalar<int32_t >(data, length); };  break;
//              case pvd::pvLong:    getData = [&](void* data, size_t& length) -> size_t { return _getScalar<int64_t >(data, length); };  break;
//              case pvd::pvUByte:   getData = [&](void* data, size_t& length) -> size_t { return _getScalar<uint8_t >(data, length); };  break;
//              case pvd::pvUShort:  getData = [&](void* data, size_t& length) -> size_t { return _getScalar<uint16_t>(data, length); };  break;
//              case pvd::pvUInt:    getData = [&](void* data, size_t& length) -> size_t { return _getScalar<uint32_t>(data, length); };  break;
//              case pvd::pvULong:   getData = [&](void* data, size_t& length) -> size_t { return _getScalar<uint64_t>(data, length); };  break;
//              case pvd::pvFloat:   getData = [&](void* data, size_t& length) -> size_t { return _getScalar<float   >(data, length); };  break;
//              case pvd::pvDouble:  getData = [&](void* data, size_t& length) -> size_t { return _getScalar<double  >(data, length); };  break;
            logging::info("scalar PV name: %s,  %s type: '%s' (%d),  length: %zd,  rank: %zd",
                          fullName.c_str(),
                          pvd::TypeFunc::name(field->getField()->getType()),
                          pvd::ScalarTypeFunc::name(type),
                          type, nelem, rank);
            break;
        }
        case pvd::scalarArray: {
            auto pvScalarArray = _strct->getSubField<pvd::PVScalarArray>(offset);
            if (!pvScalarArray)  throw "pvScalarArray is NULL";
            auto scalarArray   = pvScalarArray->getScalarArray();
            if (!scalarArray)  throw "scalarArray is NULL";
            type               = scalarArray->getElementType();
            nelem              = pvScalarArray->getLength();
// This was tested and is known to work
//          switch (scalarArray->getElementType()) {
//              case pvd::pvBoolean: getData = [&](void* data, size_t& length) -> size_t { return _getArray<uint8_t >(data, length); };  break;
//              case pvd::pvByte:    getData = [&](void* data, size_t& length) -> size_t { return _getArray<int8_t  >(data, length); };  break;
//              case pvd::pvShort:   getData = [&](void* data, size_t& length) -> size_t { return _getArray<int16_t >(data, length); };  break;
//              case pvd::pvInt:     getData = [&](void* data, size_t& length) -> size_t { return _getArray<int32_t >(data, length); };  break;
//              case pvd::pvLong:    getData = [&](void* data, size_t& length) -> size_t { return _getArray<int64_t >(data, length); };  break;
//              case pvd::pvUByte:   getData = [&](void* data, size_t& length) -> size_t { return _getArray<uint8_t >(data, length); };  break;
//              case pvd::pvUShort:  getData = [&](void* data, size_t& length) -> size_t { return _getArray<uint16_t>(data, length); };  break;
//              case pvd::pvUInt:    getData = [&](void* data, size_t& length) -> size_t { return _getArray<uint32_t>(data, length); };  break;
//              case pvd::pvULong:   getData = [&](void* data, size_t& length) -> size_t { return _getArray<uint64_t>(data, length); };  break;
//              case pvd::pvFloat:   getData = [&](void* data, size_t& length) -> size_t { return _getArray<float   >(data, length); };  break;
//              case pvd::pvDouble:  getData = [&](void* data, size_t& length) -> size_t { return _getArray<double  >(data, length); };  break;
            logging::info("scalarArray PV name: %s,  %s type: '%s' (%d),  length: %zd,  rank: %zd",
                          fullName.c_str(),
                          pvd::TypeFunc::name(field->getField()->getType()),
                          pvd::ScalarTypeFunc::name(type),
                          type, nelem, rank);
            break;
        }
         case pvd::union_: {
            auto pvUnion       = _strct->getSubField<pvd::PVUnion>(offset);
            if (!pvUnion)  throw "pvUnion is NULL";
            auto union_        = pvUnion->getUnion();
            if (!union_)  throw "union is NULL";
            auto pvScalarArray = pvUnion->get<pvd::PVScalarArray>();
            if (!pvScalarArray)  throw "Union's pvScalarArray is NULL";
            auto scalarArray   = pvScalarArray->getScalarArray();
            if (!scalarArray)  throw "Union's scalarArray is NULL";
            type               = scalarArray->getElementType();
            nelem              = pvScalarArray->getLength();
// This has NOT been tested
//          switch (scalarArray->getElementType()) {
//              case pvd::pvBoolean: getData = [&](void* data, size_t& length) -> size_t { return _getUnion<uint8_t >(data, length); };  break;
//              case pvd::pvByte:    getData = [&](void* data, size_t& length) -> size_t { return _getUnion<int8_t  >(data, length); };  break;
//              case pvd::pvShort:   getData = [&](void* data, size_t& length) -> size_t { return _getUnion<int16_t >(data, length); };  break;
//              case pvd::pvInt:     getData = [&](void* data, size_t& length) -> size_t { return _getUnion<int32_t >(data, length); };  break;
//              case pvd::pvLong:    getData = [&](void* data, size_t& length) -> size_t { return _getUnion<int64_t >(data, length); };  break;
//              case pvd::pvUByte:   getData = [&](void* data, size_t& length) -> size_t { return _getUnion<uint8_t >(data, length); };  break;
//              case pvd::pvUShort:  getData = [&](void* data, size_t& length) -> size_t { return _getUnion<uint16_t>(data, length); };  break;
//              case pvd::pvUInt:    getData = [&](void* data, size_t& length) -> size_t { return _getUnion<uint32_t>(data, length); };  break;
//              case pvd::pvULong:   getData = [&](void* data, size_t& length) -> size_t { return _getUnion<uint64_t>(data, length); };  break;
//              case pvd::pvFloat:   getData = [&](void* data, size_t& length) -> size_t { return _getUnion<float   >(data, length); };  break;
//              case pvd::pvDouble:  getData = [&](void* data, size_t& length) -> size_t { return _getUnion<double  >(data, length); };  break;
            logging::info("union PV name: %s,  %s type: '%s' (%d),  length: %zd,  rank: %zd",
                          fullName.c_str(),
                          pvd::TypeFunc::name(field->getField()->getType()),
                          pvd::ScalarTypeFunc::name(type),
                          type, nelem, rank);
            break;
        }
        default: {
            logging::warning("%s: Unsupported field type '%s'",
                             fullName.c_str(),
                             pvd::TypeFunc::name(field->getField()->getType()));
            return 1;
        }
    }

    return 0;
}

void PvMonitorBase::getTimestamp(int64_t& seconds, int32_t& nanoseconds) const
{
    // This seems to be the intended way to retrieve the timestamp
    // However there is some problem with const...
    //pvd::PVTimeStamp pvTimeStamp;
    //pvTimeStamp.attach(_strct->getSubField("timeStamp"));
    //pvd::TimeStamp ts;
    //pvTimeStamp.get(ts);
    //
    //seconds     = ts.getSecondsPastEpoch();
    //nanoseconds = ts.getNanoseconds();

    seconds     = _strct->getSubField<pvd::PVScalar>("timeStamp.secondsPastEpoch")->getAs<long>();
    nanoseconds = _strct->getSubField<pvd::PVScalar>("timeStamp.nanoseconds")->getAs<int>();
    seconds    -= m_epochDiff;
}

size_t PvMonitorBase::_getData(std::shared_ptr<const pvd::PVScalar> const& pvScalar, void* data, size_t& size)
{
    auto scalar = pvScalar->getScalar();
    switch (scalar->getScalarType()) {
        case pvd::pvBoolean: return _getScalar<uint8_t >(pvScalar, data, size);
        case pvd::pvByte:    return _getScalar<int8_t  >(pvScalar, data, size);
        case pvd::pvShort:   return _getScalar<int16_t >(pvScalar, data, size);
        case pvd::pvInt:     return _getScalar<int32_t >(pvScalar, data, size);
        case pvd::pvLong:    return _getScalar<int64_t >(pvScalar, data, size);
        case pvd::pvUByte:   return _getScalar<uint8_t >(pvScalar, data, size);
        case pvd::pvUShort:  return _getScalar<uint16_t>(pvScalar, data, size);
        case pvd::pvUInt:    return _getScalar<uint32_t>(pvScalar, data, size);
        case pvd::pvULong:   return _getScalar<uint64_t>(pvScalar, data, size);
        case pvd::pvFloat:   return _getScalar<float   >(pvScalar, data, size);
        case pvd::pvDouble:  return _getScalar<double  >(pvScalar, data, size);
        default: {
            auto fullName(name() + "." + pvScalar->getFullName());
            logging::critical("%s: Unsupported %s type %s (%d)",
                              fullName.c_str(),
                              pvd::TypeFunc::name(pvScalar->getField()->getType()),
                              pvd::ScalarTypeFunc::name(scalar->getScalarType()),
                              scalar->getScalarType());
            throw "Unsupported scalar field type";
        }
    }
    return 0;
}

size_t PvMonitorBase::_getData(std::shared_ptr<const pvd::PVScalarArray> const& pvScalarArray, void* data, size_t& size)
{
    auto scalarArray = pvScalarArray->getScalarArray();
    switch (scalarArray->getElementType()) {
        case pvd::pvBoolean: return _getArray<uint8_t >(pvScalarArray, data, size);
        case pvd::pvByte:    return _getArray<int8_t  >(pvScalarArray, data, size);
        case pvd::pvShort:   return _getArray<int16_t >(pvScalarArray, data, size);
        case pvd::pvInt:     return _getArray<int32_t >(pvScalarArray, data, size);
        case pvd::pvLong:    return _getArray<int64_t >(pvScalarArray, data, size);
        case pvd::pvUByte:   return _getArray<uint8_t >(pvScalarArray, data, size);
        case pvd::pvUShort:  return _getArray<uint16_t>(pvScalarArray, data, size);
        case pvd::pvUInt:    return _getArray<uint32_t>(pvScalarArray, data, size);
        case pvd::pvULong:   return _getArray<uint64_t>(pvScalarArray, data, size);
        case pvd::pvFloat:   return _getArray<float   >(pvScalarArray, data, size);
        case pvd::pvDouble:  return _getArray<double  >(pvScalarArray, data, size);
        default: {
            auto fullName(name() + "." + pvScalarArray->getFullName());
            logging::critical("%s: Unsupported %s type '%s' (%d)",
                              fullName.c_str(),
                              pvd::TypeFunc::name(pvScalarArray->getField()->getType()),
                              pvd::ScalarTypeFunc::name(scalarArray->getElementType()),
                              scalarArray->getElementType());
            throw "Unsupported scalarArray field type";
        }
    }
    return 0;
}

size_t PvMonitorBase::_getData(std::shared_ptr<const pvd::PVUnion> const& pvUnion, void* data, size_t& size)
{
    auto union_ = pvUnion->getUnion();
    if (pvUnion->getNumberFields() != 1) {
        auto fullName(name() + "." + pvUnion->getFullName());
        logging::error("%s: Unsupported field count %d\n",
                       fullName.c_str(), pvUnion->getNumberFields());
        throw "Unsupported union field count";
    }
    auto field = pvUnion->get()->getField();
    switch (field->getType()) {
        case pvd::scalar:
            return _getData(pvUnion->get<pvd::PVScalar>(), data, size);
        case pvd::scalarArray:
            return _getData(pvUnion->get<pvd::PVScalarArray>(), data, size);
        default: {
            auto fullName(name() + "." + pvUnion->getFullName());
            logging::error("%s: Unsupported union field type '%s'",
                           fullName.c_str(),
                           pvd::TypeFunc::name(field->getType()));
            throw "Unsupported union field type";
        }
    }
    return 0;
}

std::vector<uint32_t> PvMonitorBase::_getDimensions(size_t count)
{
    auto pvStructureArray = _strct->getSubField<pvd::PVStructureArray>("dimension");
    if (pvStructureArray) {
        auto pvStructure = pvStructureArray->view();
        std::vector<uint32_t> counts(pvStructureArray->getLength());
        for (unsigned i = 0; i < counts.size(); ++i) {
            // Revisit: EPICS data shows up in [x,y] order but psana wants [y,x]
            counts[i] = pvStructure[counts.size() - 1 - i]->getSubField<pvd::PVInt>("size")->getAs<uint32_t>();
        }
        return counts;
    }

    return std::vector<uint32_t>(1, count);
}

std::vector<uint32_t> PvMonitorBase::getData(void* data, size_t& payloadSize)
{
    auto   field  = _strct->getSubField<pvd::PVField>("value");
    auto   offset = field->getFieldOffset();
    size_t size   = payloadSize;
    size_t count;
    switch (field->getField()->getType()) {
        case pvd::scalar: {
            auto pvScalar = _strct->getSubField<pvd::PVScalar>(offset);
            count = _getData(pvScalar, data, size);
            payloadSize = size;
            return std::vector<uint32_t>(0);
        }
        case pvd::scalarArray: {
            auto pvScalarArray = _strct->getSubField<pvd::PVScalarArray>(offset);
            count = _getData(pvScalarArray, data, size);
            break;
        }
        case pvd::union_: {
            auto pvUnion = _strct->getSubField<pvd::PVUnion>(offset);
            count = _getData(pvUnion, data, size);
            break;
        }
        default: {
            auto fullName(name() + "." + field->getFullName());
            logging::error("%s: Unsupported field type '%s'",
                           fullName.c_str(),
                           pvd::TypeFunc::name(field->getField()->getType()));
            throw "Unsupported field type";
        }
    }
    payloadSize = size;
    return _getDimensions(count);
}

} // namespace Pds_Epics
