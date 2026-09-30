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
protected:
    nlohmann::json connectionInfo(const nlohmann::json& msg) override;
    void           _connectionInfo(PyObject*) override;
    unsigned       _configure(XtcData::Xtc&, const void* bufEnd, XtcData::ConfigIter&) override;
    void           _event    (XtcData::Xtc&, const void* bufEnd,
                              uint64_t l1count,
                              std::vector< XtcData::Array<uint8_t> >&) override;

private:
    void           _init_dual_dev(std::string detname,
                                  std::string data_fpga,
                                  size_t data_lane_mask);
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
    unsigned          m_nprints;
  };

}

