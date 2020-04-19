#pragma once

#include "BEBDetector.hh"

namespace Drp {

class Opal : public BEBDetector
{
public:
    Opal(Parameters* para, MemPool* pool);
    nlohmann::json connectionInfo() override;
private:
    void           _connect  (PyObject*) override;
    unsigned       _configure(XtcData::Xtc&) override;
    void           _event    (XtcData::Xtc&,
                              std::vector< XtcData::Array<uint8_t> >&) override;
private:
    XtcData::NamesId  m_evtNamesId;
    unsigned          m_rows;
    unsigned          m_columns;
  };

}
