/**
 * @file
 * @brief EpixUHR, the DRP detector class for the ePixUHR camera.
 */

#pragma once

#include "BEBDetector.hh"
#include "psdaq/service/Semaphore.hh"

/** Defined as 24 (also in EpixUHR3x2.hh); no use was found in the psdaq/drp source files. */
#define NUM_BANKS 24


namespace Drp {

/** BEBDetector for the ePixUHR. L1Accept data is a 1D byte array per ASIC (4 x 192 x 168 x 12 / 8 bytes, the code comment says 12-bit packing); the monitoring stream is switched with the Python functions `{detType}_enable` and `{detType}_disable`. */
class EpixUHR : public BEBDetector
{
public:
    /** Set virtual channel 0, initialize the BEBDetector with the detector name and install a one-shot handler for SIGINT, SIGTERM, SIGQUIT, SIGABRT and SIGSEGV that prints the signal, restores the previous handler and re-raises the signal. */
    EpixUHR(Parameters* para, MemPool* pool);
    /** Does nothing; empty body. */
    ~EpixUHR();
    /** Reset the error print budget, call monStreamDisable() and return 0. */
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

    /** Return the timing header of DMA buffer index: the data after the first event-batcher header (the descrambling code is commented out). */
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
    unsigned          m_nprints;
  };

}
