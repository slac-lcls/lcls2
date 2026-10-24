/**
 * @file
 * @brief EpixQuad, the DRP detector class for the ePix quad camera.
 */
#pragma once

#include "BEBDetector.hh"
#include "psdaq/service/Semaphore.hh"

namespace Drp {

/** BEBDetector for the ePix quad. L1Accept data comes from the image in subframe 2 of 4 expected subframes; the monitoring stream is switched with the Python functions `{detType}_enable` and `{detType}_disable`. */
class EpixQuad : public BEBDetector
{
public:
    /** Set virtual channel 0, construct the BEBDetector base and initialize it with the detector name. */
    EpixQuad(Parameters* para, MemPool* pool);
    /** Does nothing; empty body. */
    ~EpixQuad();
    /** Switch the monitoring stream off (calls `{detType}_enable` in Python) and return 0. */
    unsigned enable   (XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& info) override;
    /** Switch the monitoring stream on (calls `{detType}_disable` in Python) and return 0. */
    unsigned disable  (XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& info) override;
    /** Call Detector::slowupdate() (a variant that copies cached environment data is compiled only with SLOW_UPDATE_ENV). */
    void slowupdate(XtcData::Xtc&, const void* bufEnd) override;
    /** Return false. */
    bool scanEnabled() override;
    /** Does nothing; empty body. */
    void shutdown() override;
    /** Declared here; no definition was found in psdaq/drp. */
    void write_image(XtcData::Xtc&, const void* bufEnd, std::vector< XtcData::Array<uint8_t> >&, XtcData::NamesId&);
protected:
    void           _connectionInfo(PyObject*) override;
    unsigned       _configure(XtcData::Xtc&, const void* bufEnd, XtcData::ConfigIter&) override;
    void           _event    (XtcData::Xtc&, const void* bufEnd,
                              uint64_t l1count,
                              std::vector< XtcData::Array<uint8_t> >&) override;
private:
    void           _monStreamEnable ();
    void           _monStreamDisable();
protected:
    Pds::Semaphore    m_env_sem;
    bool              m_env_empty;
    XtcData::NamesId  m_evtNamesId[8];
  };

}
