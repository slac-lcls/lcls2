#pragma once

#include "BEBDetector.hh"
#include "psdaq/service/Collection.hh"

namespace Drp {

namespace Piranha {
    class TT;
    class TTSim;
    class TTSimL1;
    class TTSimL2;
};

class Piranha4 : public BEBDetector
{
public:
    Piranha4(Parameters* para, MemPool* pool);
    ~Piranha4();
    void slowupdate(XtcData::Xtc&, const void* bufEnd) override;
    void shutdown() override;
    void write_image(XtcData::Xtc&, const void* bufEnd, std::vector< XtcData::Array<uint8_t> >&, XtcData::NamesId&);
protected:
    void           _connectionInfo(PyObject*) override;
    unsigned       _configure(XtcData::Xtc&, const void* bufEnd, XtcData::ConfigIter&) override;
    void           _event    (XtcData::Xtc&, const void* bufEnd,
                              std::vector< XtcData::Array<uint8_t> >&) override;
    void           _fatal_error(std::string errMsg);
protected:
    friend class Piranha::TT;
    friend class Piranha::TTSimL1;
    friend class Piranha::TTSimL2;

    XtcData::NamesId  m_evtNamesId;
    unsigned          m_pixels;

    Piranha::TT*      m_tt;
    Piranha::TTSim*   m_sim;
private:
    ZmqContext m_context;
    ZmqSocket m_notifySocket;
  };

}
