/**
 * @file
 * @brief TimingBEB, a BEBDetector that records timing data from an event-batcher stream.
 */
#pragma once

#include "BEBDetector.hh"
#include "psdaq/service/Semaphore.hh"

namespace Drp {

/** BEBDetector whose L1Accept data is TimingDef data written by TimingDef::createDataETM() from the first subframe (header) and the last subframe (ETM record) of each event. */
class TimingBEB : public BEBDetector
{
public:
    /** Construct the BEBDetector base and initialize it with _init(detName) and _init_feb(). */
    TimingBEB(Parameters* para, MemPool* pool);
    /** Does nothing; empty body. */
    ~TimingBEB();
    /** Call BEBDetector::connect (keeps the connect JSON and readout group), then call the Python function ts_connect with the connect JSON string. */
    void connect(const nlohmann::json&, const std::string&) override;
    /** Return true. */
    bool scanEnabled() override;
    /** Does nothing; empty body (BEBDetector::shutdown is not called). */
    void shutdown() override;
protected:
    void           _connectionInfo(PyObject*) override;
    unsigned       _configure(XtcData::Xtc&, const void* bufEnd, XtcData::ConfigIter&) override;
    void           _event    (XtcData::Xtc&, const void* bufEnd,
                              uint64_t l1count,
                              std::vector< XtcData::Array<uint8_t> >&) override;
protected:
    XtcData::NamesId  m_evtNamesId;
  };

}
