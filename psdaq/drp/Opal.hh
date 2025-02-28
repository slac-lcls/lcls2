#pragma once

#include "BEBDetector.hh"
#include "psdaq/service/Collection.hh"

namespace Drp {

class OpalTT;
class OpalTTSim;

class Opal : public BEBDetector
{
public:
    Opal(Parameters* para, MemPool* pool);
    ~Opal();
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
    friend class OpalTT;
    friend class OpalTTSimL1;
    friend class OpalTTSimL2;

    XtcData::NamesId  m_evtNamesId;
    unsigned          m_rows;
    unsigned          m_columns;

    OpalTT*           m_tt;
    OpalTTSim*        m_sim;
private:
    ZmqContext m_context;
    ZmqSocket m_notifySocket;
  };

}
