/**
 * @file
 * @brief Epix100, the DRP detector class for the epix100 camera.
 */
#pragma once

#include "BEBDetector.hh"
#include "psdaq/service/Semaphore.hh"

namespace Drp {

/** BEBDetector for the epix100. L1Accept data is a 704 x 768 UINT16 frame unscrambled from subframe 3; the monitoring stream is switched with the Python functions `{detType}_enable` and `{detType}_disable`. */
class Epix100 : public BEBDetector
{
public:
    /** Construct the BEBDetector base, initialize it with the detector name and install a one-shot handler for SIGINT, SIGTERM, SIGQUIT, SIGABRT and SIGSEGV that prints the signal, restores the previous handler and re-raises the signal. */
    Epix100(Parameters* para, MemPool* pool);
    /** Does nothing; empty body. */
    ~Epix100();
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
protected:
    void           _connectionInfo(PyObject*) override;
    unsigned       _configure(XtcData::Xtc&, const void* bufEnd, XtcData::ConfigIter&) override;
    void           _event    (XtcData::Xtc&, const void* bufEnd,
                              uint64_t l1count,
                              std::vector< XtcData::Array<uint8_t> >&) override;
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
  };

}
