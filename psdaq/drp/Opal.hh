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
    nlohmann::json connectionInfo() override;
    void slowupdate(XtcData::Xtc&) override;
    void shutdown() override;
    void write_image(XtcData::Xtc&, std::vector< XtcData::Array<uint8_t> >&, XtcData::NamesId&);
protected:
    void           _connect  (PyObject*) override;
    unsigned       _configure(XtcData::Xtc&, XtcData::ConfigIter&) override;
    void           _event    (XtcData::Xtc&,
                              std::vector< XtcData::Array<uint8_t> >&) override;
    void           _fatal_error(std::string errMsg);
protected:
    friend class OpalTT;
    friend class OpalTTSim;

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
