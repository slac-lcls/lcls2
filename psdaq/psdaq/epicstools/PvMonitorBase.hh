/**
 * @file
 * @brief PvMonitorBase, a MonTracker that copies the monitored field of a PV into raw payload buffers.
 */
#pragma once

#include <string>
#include <vector>
#include "MonTracker.hh"

#include "xtcdata/xtc/Array.hh"         // For MaxRank

#include "epicsTime.h"                  // POSIX_TIME_AT_EPICS_EPOCH

namespace Pds_Epics {

/** MonTracker that copies the payload field of a PV (value by default) into raw buffers: getParams() inspects the field's type and selects the getData function. */
class PvMonitorBase : public MonTracker
{
public:
    /** Monitor pvName through provider (pva by default) with request; fieldName names the payload field. */
    PvMonitorBase(const std::string& pvName,
                  const std::string& provider  = "pva",
                  const std::string& request   = "field()",
                  const std::string& fieldName = "value") :
      MonTracker(provider, pvName, request),
      m_pvaProvider(provider == "pva"),
      m_fieldName(fieldName)
    {
    }
    /** Does nothing; empty virtual destructor. */
    virtual ~PvMonitorBase() {}
public:
    /** Log the name, offset and type of each top-level field of the latest structure. Returns 1 if a needed part is missing, else 0. */
    int printStructure() const;
    /** From the latest structure, return the element type, element count and rank of the payload field and set getData for it. Strings count MAX_STRING_SIZE bytes each and enum_t structures are treated as strings. Returns 1 if the structure or a needed part of it is missing, else 0; throws a C string for a missing payload or an unsupported type. */
    int getParams(pvd::ScalarType& type, size_t& size, size_t& rank);
    /** Return true if the provider is pva. */
    bool isPva() { return  m_pvaProvider; }
    /** Return true if the provider is not pva. */
    bool isCa()  { return !m_pvaProvider; }
    /** Return the seconds and nanoseconds of the timeStamp field of the latest structure. */
    void getTimestamp(int64_t& seconds, int32_t& nanoseconds) {
        nanoseconds = _strct->getSubField<pvd::PVScalar>("timeStamp.nanoseconds")->getAs<int>();
        seconds     = _strct->getSubField<pvd::PVScalar>("timeStamp.secondsPastEpoch")->getAs<long>();
    }
    /** Same as getTimestamp(): the secondsPastEpoch value is returned unchanged. */
    void getTimestampPosix(int64_t& seconds, int32_t& nanoseconds) {
        getTimestamp(seconds, nanoseconds);
    }
    /** Like getTimestampPosix(), with POSIX_TIME_AT_EPICS_EPOCH subtracted from the seconds. */
    void getTimestampEpics(int64_t& seconds, int32_t& nanoseconds) {
        getTimestampPosix(seconds, nanoseconds);
        seconds -= POSIX_TIME_AT_EPICS_EPOCH;
    }
public:
    /** Rank limit for the getData shape. */
    enum { MaxRank = XtcData::MaxRank  /**< XtcData::MaxRank. */ };
    // For getData functions:
    // data:    A pointer to the payload buffer to be filled
    // size:    The available size in the payload buffer for filling
    // shape:   The shape of the data (ignored if rank is zero)
    // returns: The amount of payload space actually used, or could have been
    //          used, by the data in lieu of truncation
    //          (if returned size > avalable size, truncation occurred)
    /** Copy the payload field into data (size bytes available) and fill shape; returns the bytes used, or the bytes that would have been used if more than size (truncation), per the code comment. Set by getParams(). */
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
    const bool        m_pvaProvider;
    const std::string m_fieldName;
};

}
