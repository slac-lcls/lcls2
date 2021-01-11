#ifndef EPICS_MONITOR_PV_HH
#define EPICS_MONITOR_PV_HH

#include <vector>
#include <string>
#include <mutex>
#include <condition_variable>

#include "psdaq/epicstools/PvMonitorBase.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/NamesLookup.hh"

#include "EpicsXtcSettings.hh"


namespace Drp
{

  class EpicsMonitorPv : public Pds_Epics::PvMonitorBase
  {
  public:
    EpicsMonitorPv(const std::string& sPvName,
                   const std::string& sPvDescription,
                   const std::string& sProvider,
                   const std::string& sRequest,
                   bool               bDebug);
     ~EpicsMonitorPv();   // non-virtual destructor: this class is not for inheritance

    int  release();
    int  printPv() const;
    int  addVarDef(EpicsArchDef& varDef, size_t& size);
    int  addToXtc(XtcData::Damage& damage, bool& stale, char *pcXtcMem, size_t& iSizeXtc, std::vector<uint32_t>& sShape);

    /* Get & Set functions */
    const std::string  getPvName()         const {return name();}
    const std::string& getPvDescription()  const {return _sPvDescription;}
    bool               isConnected()       const {return _connected;}
    void               disable()                 {_bDisabled = true;}
    bool               isDisabled()        const {return _bDisabled;}
  private:
    void onConnect()    override;
    void onDisconnect() override;
    void updated()      override;
  private:
    enum State { NotReady, Ready };
  private: // Marked static to share one mutex & condvar across all instances
    static std::mutex              _mutex;
    static std::condition_variable _condition;
  private:
    std::string                    _sPvDescription;
    std::vector<uint8_t>           _pData;
    std::vector<uint32_t>          _shape;
    size_t                         _size;
    const std::string              _pvField;
    pvd::ScalarType                _type;
    size_t                         _nelem;
    size_t                         _rank;
    State                          _state;
    bool                           _bUpdated;
    bool                           _bDisabled;
    bool                           _bDebug;
  };

  typedef std::vector < std::shared_ptr<EpicsMonitorPv> > TEpicsMonitorPvList;
}       // namespace Drp

#endif
