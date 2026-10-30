#pragma once

#include "BEBDetector.hh"
#include "psdaq/service/Semaphore.hh"

#define NUM_BANKS 24


namespace Drp {

class EpixUHR3x2 : public BEBDetector
{
public:
    EpixUHR3x2(Parameters* para, MemPool* pool);
    ~EpixUHR3x2();
    unsigned enable   (XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& info) override;
    unsigned disable  (XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& info) override;
    void slowupdate(XtcData::Xtc&, const void* bufEnd) override;
    bool scanEnabled() override;
    void shutdown() override;
    void write_image(XtcData::Xtc&, const void* bufEnd, std::vector< XtcData::Array<uint8_t> >&, XtcData::NamesId&);

    Pds::TimingHeader* getTimingHeader(uint32_t index) const override;
    // The raw array's alg version, which selects psana's interpretation of it
    void setRawAlgVersion(uint8_t major, uint8_t minor, uint8_t micro)
    { m_rawAlgVersion[0] = major;  m_rawAlgVersion[1] = minor;  m_rawAlgVersion[2] = micro; }
protected:
    nlohmann::json connectionInfo(const nlohmann::json& msg) override;
    void           _connectionInfo(PyObject*) override;
    unsigned       _configure(XtcData::Xtc&, const void* bufEnd, XtcData::ConfigIter&) override;
    void           _event    (XtcData::Xtc&, const void* bufEnd,
                              uint64_t l1count,
                              std::vector< XtcData::Array<uint8_t> >&) override;

private:
    void           _init_data_path_or_emu();
    void           __event   (XtcData::Xtc&, const void* bufEnd,
                              std::vector< XtcData::Array<uint8_t> >&);
public:
    void           monStreamEnable ();
    void           monStreamDisable();
protected:
    Pds::Semaphore    m_env_sem;
    bool              m_env_empty;
    XtcData::NamesId  m_evtNamesId[2];
    unsigned          m_asics;
    bool              m_descramble;
    bool              m_monStreamOn;    // Tracks the monitor stream's state
    unsigned          m_nprints;
    uint8_t           m_rawAlgVersion[3] { 1, 0, 0 };
  };

}

