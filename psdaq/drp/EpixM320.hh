
#pragma once

#include "BEBDetector.hh"
#include "psdaq/service/Semaphore.hh"

#define NUM_BANKS 24


namespace Drp {

class EpixM320 : public BEBDetector
{
public:
    EpixM320(Parameters* para, MemPool* pool);
    ~EpixM320();
    unsigned enable   (XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& info) override;
    unsigned disable  (XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& info) override;
    void slowupdate(XtcData::Xtc&, const void* bufEnd) override;
    bool scanEnabled() override;
    void shutdown() override;
    void write_image(XtcData::Xtc&, const void* bufEnd, std::vector< XtcData::Array<uint8_t> >&, XtcData::NamesId&);

    Pds::TimingHeader* getTimingHeader(uint32_t index) const override;
protected:
    void           _connectionInfo(PyObject*) override;
    unsigned       _configure(XtcData::Xtc&, const void* bufEnd, XtcData::ConfigIter&) override;
    void           _event    (XtcData::Xtc&, const void* bufEnd,
                              std::vector< XtcData::Array<uint8_t> >&) override;
private:
    void           __event   (XtcData::Xtc&, const void* bufEnd,
                              std::vector< XtcData::Array<uint8_t> >&);
    void           _descramble(uint16_t input[3073][NUM_BANKS], uint16_t output[3073][NUM_BANKS]);
public:
    void           monStreamEnable ();
    void           monStreamDisable();
protected:
    Pds::Semaphore    m_env_sem;
    bool              m_env_empty;
    XtcData::NamesId  m_evtNamesId[2];
    unsigned          m_asics;
    bool              m_descramble;
  };

}
