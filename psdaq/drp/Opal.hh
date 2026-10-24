/**
 * @file
 * @brief Opal, the DRP detector class for the Opal camera.
 */
#pragma once

#include "BEBDetector.hh"
#include "psdaq/service/Collection.hh"

namespace Drp {

class OpalTT;
class OpalTTSim;

/** BEBDetector for the Opal camera: records an m_rows x m_columns UINT16 image, with optional timetool processing (OpalTT) and simulation from XTC files. */
class Opal : public BEBDetector
{
public:
    /** Connect the error PUSH socket to the collection server, initialize the BEBDetector with _init(detName) and _init_feb(), and create a simulator if the simxtc (OpalTTSimL1) or simxtc2 (OpalTTSimL2, with simtime) kwarg is set. */
    Opal(Parameters* para, MemPool* pool);
    /** Delete the simulator and the timetool object if present. */
    ~Opal();
    /** If timetool processing is set up (m_tt), call OpalTT::slowupdate; otherwise call Detector::slowupdate, which resets xtc to an empty Parent Xtc. */
    void slowupdate(XtcData::Xtc&, const void* bufEnd) override;
    /** Call OpalTT::shutdown if timetool processing is set up, then BEBDetector::shutdown, which calls the Python unconfig function for the detector type. */
    void shutdown() override;
    /** Add the image (m_rows x m_columns UINT16) under namesId, copying it from subframes[2]. */
    void write_image(XtcData::Xtc&, const void* bufEnd, std::vector< XtcData::Array<uint8_t> >&, XtcData::NamesId&);
protected:
    void           _connectionInfo(PyObject*) override;
    unsigned       _configure(XtcData::Xtc&, const void* bufEnd, XtcData::ConfigIter&) override;
    void           _event    (XtcData::Xtc&, const void* bufEnd,
                              uint64_t l1count,
                              std::vector< XtcData::Array<uint8_t> >&) override;
    void           _fatal_error(std::string errMsg);
protected:
    friend class OpalTT;
    friend class OpalTTSimL1;
    friend class OpalTTSimL2;

    XtcData::NamesId  m_evtNamesId;
    unsigned          m_rows;
    unsigned          m_columns;

    OpalTT*           m_tt;
    OpalTTSim*        m_sim;
private:
    ZmqContext m_context;
    ZmqSocket m_notifySocket;
  };

}
