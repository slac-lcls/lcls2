#pragma once

#include <string>
#include <vector>
#include "MonTracker.hh"

#include "xtcdata/xtc/Array.hh"         // For MaxRank

#include "epicsTime.h"                  // POSIX_TIME_AT_EPICS_EPOCH

namespace Pds_Epics {

class PvMonitorBase : public MonTracker
{
public:
    PvMonitorBase(const std::string& pvName,
                  const std::string& provider  = "pva",
                  const std::string& request   = "field()",
                  const std::string& fieldName = "value") :
      MonTracker(provider, pvName, request),
      m_epochDiff(provider == "pva" ? 0 : POSIX_TIME_AT_EPICS_EPOCH),
      m_fieldName(fieldName)
    {
    }
    virtual ~PvMonitorBase() {}
public:
    int printStructure() const;
    int getParams(pvd::ScalarType& type, size_t& size, size_t& rank);
    void getTimestamp(int64_t& seconds, int32_t& nanoseconds) {
        seconds     = _strct->getSubField<pvd::PVScalar>("timeStamp.secondsPastEpoch")->getAs<long>();
        nanoseconds = _strct->getSubField<pvd::PVScalar>("timeStamp.nanoseconds")->getAs<int>();
        seconds    -= m_epochDiff;
    }
public:
    enum { MaxRank = XtcData::MaxRank };
    // For getData functions:
    // data:    A pointer to the payload buffer to be filled
    // size:    The available size in the payload buffer for filling
    // shape:   The shape of the data (ignored if rank is zero)
    // returns: The amount of payload space actually used, or could have been
    //          used, by the data in lieu of truncation
    //          (if returned size > avalable size, truncation occurred)
    std::function<size_t(void* data, size_t size, uint32_t shape[MaxRank])> getData;
private:
    void _getDimensions(uint32_t shape[MaxRank]) const;
private:
    template<typename T>
    size_t _getScalar(std::shared_ptr<const pvd::PVScalar> const& pvScalar, void* data, size_t size, uint32_t shape[MaxRank]) const;
    template<typename T>
    size_t _getScalar(void* data, size_t size, uint32_t shape[MaxRank]) const {
        return _getScalar<T>(_strct->getSubField<pvd::PVScalar>(m_fieldName), data, size, shape);
    }
    size_t _getScalarString(void* data, size_t size, uint32_t shape[MaxRank]) const;
    template<typename T>
    size_t _getArray(std::shared_ptr<const pvd::PVScalarArray> const& pvScalarArray, void* data, size_t size, uint32_t shape[MaxRank]) const;
    template<typename T>
    size_t _getArray(void* data, size_t size, uint32_t shape[MaxRank]) const {
        return _getArray<T>(_strct->getSubField<pvd::PVScalarArray>(m_fieldName), data, size, shape);
    }
    size_t _getArrayString(void* data, size_t size, uint32_t shape[MaxRank]) const;
    size_t _getEnum(void* data, size_t size, uint32_t shape[MaxRank]) const;
    template<typename T>
    size_t _getUnionScl(void* data, size_t size, uint32_t shape[MaxRank]) const {
        const auto& pvUnion = _strct->getSubField<pvd::PVUnion>(m_fieldName);
        return _getScalar<T>(pvUnion->get<pvd::PVScalar>(), data, size, shape);
    }
    template<typename T>
    size_t _getUnionSclArr(void* data, size_t size, uint32_t shape[MaxRank]) const {
        const auto& pvUnion = _strct->getSubField<pvd::PVUnion>(m_fieldName);
        auto sz = _getArray<T>(pvUnion->get<pvd::PVScalarArray>(), data, size, shape);
        _getDimensions(shape);
        return sz;
    }
protected:
    const unsigned long m_epochDiff;
    const std::string   m_fieldName;
};

}
