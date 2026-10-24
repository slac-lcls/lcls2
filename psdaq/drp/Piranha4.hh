/**
 * @file
 * @brief Piranha4, the DRP detector class for the Piranha4 camera, which records a one-dimensional image.
 */
#pragma once

#include "BEBDetector.hh"
#include "psdaq/service/Collection.hh"

namespace Drp {

/** Timetool helper classes for Piranha4 (TT, TTSim, TTSimL1, TTSimL2), defined in Piranha4.cc and only forward-declared here. */
namespace Piranha {
    class TT;
    class TTSim;
    class TTSimL1;
    class TTSimL2;
};

/** BEBDetector for the Piranha4 camera: records a one-dimensional UINT16 image of m_pixels values, with optional timetool processing (Piranha::TT) and simulation from XTC files. */
class Piranha4 : public BEBDetector
{
public:
    /** Connect the error PUSH socket to the collection server, initialize the BEBDetector with _init(detName) and _init_feb(), and create a simulator if the simxtc (TTSimL1) or simxtc2 (TTSimL2, with simtime) kwarg is set. */
    Piranha4(Parameters* para, MemPool* pool);
    /** Delete the simulator and the timetool object if present. */
    ~Piranha4();
    /** If timetool processing is set up (m_tt), call Piranha::TT::slowupdate; otherwise call Detector::slowupdate, which resets xtc to an empty Parent Xtc. */
    void slowupdate(XtcData::Xtc&, const void* bufEnd) override;
    /** Call Piranha::TT::shutdown if timetool processing is set up, then BEBDetector::shutdown, which calls the Python unconfig function for the detector type. */
    void shutdown() override;
    /** Add the image (m_pixels UINT16 values) under namesId, copying it from subframes[2]. */
    void write_image(XtcData::Xtc&, const void* bufEnd, std::vector< XtcData::Array<uint8_t> >&, XtcData::NamesId&);
protected:
    void           _connectionInfo(PyObject*) override;
    unsigned       _configure(XtcData::Xtc&, const void* bufEnd, XtcData::ConfigIter&) override;
    void           _event    (XtcData::Xtc&, const void* bufEnd,
                              uint64_t l1count,
                              std::vector< XtcData::Array<uint8_t> >&) override;
    void           _fatal_error(std::string errMsg);
protected:
    /** Lets Piranha::TT use the protected members. */
    friend class Piranha::TT;
    /** Lets Piranha::TTSimL1 use the protected members. */
    friend class Piranha::TTSimL1;
    /** Lets Piranha::TTSimL2 use the protected members. */
    friend class Piranha::TTSimL2;

    XtcData::NamesId  m_evtNamesId;
    unsigned          m_pixels;

    Piranha::TT*      m_tt;
    Piranha::TTSim*   m_sim;
private:
    ZmqContext m_context;
    ZmqSocket m_notifySocket;
  };

}
