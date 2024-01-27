#pragma once

#include "BEBDetector.hh"
#include "psdaq/service/Semaphore.hh"

namespace Drp {

class TimingBEB : public BEBDetector
{
public:
    TimingBEB(Parameters* para, MemPool* pool);
    ~TimingBEB();
    void connect(const nlohmann::json&, const std::string&) override;
    bool scanEnabled() override;
    void shutdown() override;
protected:
    void           _connectionInfo(PyObject*) override;
    unsigned       _configure(XtcData::Xtc&, const void* bufEnd, XtcData::ConfigIter&) override;
    void           _event    (XtcData::Xtc&, const void* bufEnd,
                              std::vector< XtcData::Array<uint8_t> >&) override;
protected:
    XtcData::NamesId  m_evtNamesId;
  };

}
