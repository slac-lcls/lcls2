
#pragma once

#include "BEBDetector.hh"
#include "psdaq/service/Semaphore.hh"

namespace Drp {

class EpixM320 : public BEBDetector
{
public:
    EpixM320(Parameters* para, MemPool* pool);
    ~EpixM320();
    static const unsigned NumAsics    {   4 };
    static const unsigned NumBanks    {  24 };
    static const unsigned BankRows    {   4 };
    static const unsigned BankCols    {   6 };
    static const unsigned ElemRows    { 192 };
    static const unsigned ElemRowSize { 384 };
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
    void           _descramble(uint16_t* dst, const uint16_t* src) const;
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
