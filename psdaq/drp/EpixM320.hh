/**
 * @file
 * @brief EpixM320, the DRP detector class for the ePixM320 camera.
 */

#pragma once

#include "BEBDetector.hh"
#include "psdaq/service/Semaphore.hh"

namespace Drp {

/** BEBDetector for the ePixM320. L1Accept data holds, for each of NumAsics ASICs, a header, a descrambled ElemRows x ElemRowSize UINT16 frame and a trailer; disabled or bad ASICs are zero-filled. */
class EpixM320 : public BEBDetector
{
public:
    /** Set virtual channel 0, initialize the BEBDetector with the detector name, enable descrambling and install a one-shot handler for SIGINT, SIGTERM, SIGQUIT, SIGABRT and SIGSEGV that prints the signal, restores the previous handler and re-raises the signal. */
    EpixM320(Parameters* para, MemPool* pool);
    /** Does nothing; empty body. */
    ~EpixM320();
    static const unsigned NumAsics    {   4 };  ///< Number of ASICs (4); first dimension of the event arrays.
    static const unsigned NumBanks    {  24 };  ///< Number of banks (24) interleaved in the raw ASIC data; used by the descrambler.
    static const unsigned BankRows    {   4 };  ///< Bank rows (4) of a descrambled ASIC frame.
    static const unsigned BankCols    {   6 };  ///< Bank columns (6) of a descrambled ASIC frame.
    static const unsigned ElemRows    { 192 };  ///< Rows per ASIC frame (192).
    static const unsigned ElemRowSize { 384 };  ///< Pixels per row of an ASIC frame (384).
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

    /** Return the timing header of DMA buffer index, found after the first event-batcher header. With descrambling enabled and words 2 and 3 zero, the word pairs at offsets 4, 8, 12 and 16 are first moved in place to offsets 2, 4, 6 and 8. */
    Pds::TimingHeader* getTimingHeader(uint32_t index) const override;
protected:
    void           _connectionInfo(PyObject*) override;
    unsigned       _configure(XtcData::Xtc&, const void* bufEnd, XtcData::ConfigIter&) override;
    void           _event    (XtcData::Xtc&, const void* bufEnd,
                              uint64_t l1count,
                              std::vector< XtcData::Array<uint8_t> >&) override;
private:
    void           _descramble(uint16_t* dst, const uint16_t* src) const;
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
    bool              m_logOnce;
  };

}
