/**
 * @file
 * @brief EpixHR2x2, the DRP detector class for the ePixHR 2x2 camera.
 */
#pragma once

#include "BEBDetector.hh"
#include "psdaq/service/Semaphore.hh"

namespace Drp {

/** BEBDetector for the ePixHR 2x2. L1Accept data is a 288 x 384 UINT16 raw frame assembled from subframes 3 and 4; the monitoring stream is switched with the Python functions `{detType}_enable` and `{detType}_disable`. */
class EpixHR2x2 : public BEBDetector
{
public:
    /** Construct the BEBDetector base, initialize it with the detector name, disable descrambling and install a one-shot handler for SIGINT, SIGTERM, SIGQUIT, SIGABRT and SIGSEGV that prints the signal, restores the previous handler and re-raises the signal. */
    EpixHR2x2(Parameters* para, MemPool* pool);
    /** Does nothing; empty body. */
    ~EpixHR2x2();
    /** Call monStreamDisable() and return 0. */
    unsigned enable   (XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& info) override;
    /** Call monStreamEnable() and return 0. */
    unsigned disable  (XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& info) override;
    /** Call Detector::slowupdate(). */
    void slowupdate(XtcData::Xtc&, const void* bufEnd) override;
    /** Return false. */
    bool scanEnabled() override;
    /** If the configuration module is loaded and the monitor stream is off (monStreamDisable() was called last), call monStreamEnable(). */
    void shutdown() override;
    /** Declared here; no definition was found in psdaq/drp. */
    void write_image(XtcData::Xtc&, const void* bufEnd, std::vector< XtcData::Array<uint8_t> >&, XtcData::NamesId&);

    /** Return the timing header of DMA buffer index, located after two nested event-batcher headers. With descrambling enabled and words 2 and 3 zero, the word pairs at offsets 4, 8, 12 and 16 are first moved in place to offsets 2, 4, 6 and 8 (descrambling is disabled by the constructor). */
    Pds::TimingHeader* getTimingHeader(uint32_t index) const override;
protected:
    void           _connectionInfo(PyObject*) override;
    unsigned       _configure(XtcData::Xtc&, const void* bufEnd, XtcData::ConfigIter&) override;
    void           _event    (XtcData::Xtc&, const void* bufEnd,
                              uint64_t l1count,
                              std::vector< XtcData::Array<uint8_t> >&) override;
private:
    void           __event   (XtcData::Xtc&, const void* bufEnd,
                              std::vector< XtcData::Array<uint8_t> >&);
public:
    /** Call the Python function `{detType}_disable` of the configuration module with the device root object. */
    void           monStreamEnable ();
    /** Call the Python function `{detType}_enable` of the configuration module with the device root object. */
    void           monStreamDisable();
protected:
    Pds::Semaphore    m_env_sem;
    bool              m_env_empty;
    bool              m_monStreamOn;
    XtcData::NamesId  m_evtNamesId[2];
    unsigned          m_asics;
    bool              m_descramble;
  };

}
