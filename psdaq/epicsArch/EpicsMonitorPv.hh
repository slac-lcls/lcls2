#ifndef EPICS_MONITOR_PV_HH
#define EPICS_MONITOR_PV_HH

#include <vector>
#include <string>

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
                   bool               bDebug);
     ~EpicsMonitorPv();   // non-virtual destructor: this class is not for inheritance

    int  release();
    int  printPv() const;
    int  addDef(EpicsArchDef& def, size_t& size);
    int  addToXtc(bool& stale, char *pcXtcMem, size_t& iSizeXtc, std::vector<uint32_t>& sShape);

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
    std::string           _sPvDescription;
    std::vector<uint8_t>  _pData;
    std::vector<uint32_t> _shape;
    size_t                _size;
    bool                  _bUpdated;
    bool                  _bDisabled;
    bool                  _bDebug;
  };

  typedef std::vector < std::shared_ptr<EpicsMonitorPv> > TEpicsMonitorPvList;
}       // namespace Drp

#endif
