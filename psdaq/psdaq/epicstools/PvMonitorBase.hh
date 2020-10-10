#pragma once

#include <string>
#include <vector>
#include "MonTracker.hh"

#include "epicsTime.h"                  // POSIX_TIME_AT_EPICS_EPOCH

namespace Pds_Epics {

class PvMonitorBase : public MonTracker
{
public:
    PvMonitorBase(const std::string& channelName,
                  const std::string& provider = "pva") :
      MonTracker(provider, channelName),
      m_epochDiff(provider == "pva" ? 0 : POSIX_TIME_AT_EPICS_EPOCH)
    {
    }
  virtual ~PvMonitorBase() {}
public:
    void printStructure() const;
    bool ready(const std::string& request, unsigned tmo)
    {
        return getComplete(request, tmo);
    }
    int getParams(const std::string& name,
                  pvd::ScalarType&   type,
                  size_t&            size,
                  size_t&            rank);
    void getTimestamp(int64_t& seconds, int32_t& nanoseconds) const;
    std::vector<uint32_t> getData(void* data, size_t& length);
private:
    size_t _getData(std::shared_ptr<const pvd::PVScalar>      const& pvScalar,      void* data, size_t& length);
    size_t _getData(std::shared_ptr<const pvd::PVScalarArray> const& pvScalarArray, void* data, size_t& length);
    size_t _getData(std::shared_ptr<const pvd::PVUnion>       const& pvUnion,       void* data, size_t& length);
    std::vector<uint32_t> _getDimensions(size_t length);
private:
//  public:
//    std::function<size_t(void* data, size_t& size)> getData;
//  private:
//    template<typename T> size_t _getScalar(void* data, size_t& size) {
//      *static_cast<T*>(data) = _strct->getSubField<pvd::PVScalar>("value")->getAs<T>();
//      size = sizeof(T);
//      return 1;
//    }
//    template<typename T> size_t _getArray(void* data, size_t& size) {
//      //pvd::shared_vector<const T> vec((T*)data, [](void*){}, 0, 128); // Doesn't work
//      pvd::shared_vector<const T> vec;
//      _strct->getSubField<pvd::PVScalarArray>("value")->getAs<T>(vec);
//      auto count = vec.size();
//      auto sz    = count * sizeof(T);
//      memcpy(data, vec.data(), sz < size ? sz : size); // Possibly truncates
//      size = sz;                                       // Return desired size
//      return count;
//    }
  // The above are tested, but the following hasn't been
//    template<typename T> size_t _getUnion(void* data, size_t& size) {
//      //pvd::shared_vector<const T> vec((T*)data, [](void*){}, 0, 128); // Doesn't work
//      pvd::shared_vector<const T> vec;
//      _strct->getSubField<pvd::PVUnion>("value")->get<pvd::PVScalarArray>()->getAs<T>(vec);
//      auto count = vec.size();
//      auto sz    = count * sizeof(T);
//      memcpy(data, vec.data(), sz < size ? sz : size); // Possibly truncates
//      size = sz;                                       // Return desired size
//      return count;
//    }
    template<typename T> size_t _getScalar(std::shared_ptr<const pvd::PVScalar> const& pvScalar, void* data, size_t& size) {
        *static_cast<T*>(data) = pvScalar->getAs<T>();
        size = sizeof(T);
        return 1;
    }
    template<typename T> size_t _getArray(std::shared_ptr<const pvd::PVScalarArray> const& pvScalarArray, void* data, size_t& size) {
        //pvd::shared_vector<const T> vec((T*)data, [](void*){}, 0, 128); // Doesn't work
        pvd::shared_vector<const T> vec;
        pvScalarArray->getAs<T>(vec);
        auto count = vec.size();
        auto sz    = count * sizeof(T);
        memcpy(data, vec.data(), sz < size ? sz : size); // Possibly truncates
        size = sz;                                       // Return desired size
        return count;
    }
protected:
    const unsigned long m_epochDiff;
};

}
