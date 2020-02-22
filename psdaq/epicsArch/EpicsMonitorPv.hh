#ifndef EPICS_MONITOR_PV_HH
#define EPICS_MONITOR_PV_HH

#include <vector>
#include <string>

#include "psdaq/epicstools/MonTracker.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/NamesLookup.hh"

#include "EpicsXtcSettings.hh"


namespace Pds
{

  class EpicsMonitorPv : public Pds_Epics::MonTracker
  {
  public:
    EpicsMonitorPv(const std::string& sPvName,
                   const std::string& sPvDescription, bool bProviderType);
     ~EpicsMonitorPv();   // non-virtual destructor: this class is not for inheritance

    int  release();
    bool ready(const std::string& request);
    int  printPv() const;
    int  addDef(EpicsArchDef& def, size_t& payloadSize);
    void addNames(XtcData::Xtc& xtc, XtcData::NamesLookup& namesLookup, unsigned nodeId);
    int  addToXtc(bool& stale, char *pcXtcMem, size_t& iSizeXtc, size_t& iLength);

    /* Get & Set functions */
    const std::string  getPvName()         const {return _name;}
    const std::string& getPvDescription()  const {return _sPvDescription;}
    bool               isConnected()       const {return _connected;}

  public:
    std::function<size_t(void* data, size_t& length)> getData;
  private:
    template<typename T> size_t _getDatumT(void* data, size_t& length) {
      *static_cast<T*>(data) = _strct->getSubField<pvd::PVScalar>("value")->getAs<T>();
      length = 1;
      return sizeof(T);
    }
    template<typename T> size_t _getDataT(void* data, size_t& length) {
      //pvd::shared_vector<const T> vec((T*)data, [](void*){}, 0, 128); // Doesn't work
      pvd::shared_vector<const T> vec;
      _strct->getSubField<pvd::PVScalarArray>("value")->getAs<T>(vec);
      length = vec.size();
      size_t size = length * sizeof(T);
      memcpy(data, vec.data(), size);
      return size;
    }
  private:
    void updated()   override;
  private:
    std::string _sPvDescription;
    bool        _bProviderType;
    bool        _bUpdated;
    void*       _pData;
    size_t      _size;
    size_t      _length;
  };

  typedef std::vector < std::shared_ptr<EpicsMonitorPv> > TEpicsMonitorPvList;
}       // namespace Pds

#endif
