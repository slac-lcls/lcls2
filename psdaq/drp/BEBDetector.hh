/**
 * @file
 * @brief BEBDetector, the base of DRP detectors whose events are built by the AxiStreamBatcherEventBuilder firmware and configured through Python, and its NamesId index constants.
 */
#pragma once
/**
 **   This class services detectors whose event data is formatted from the AxiStreamBatcherEventBuilder firmware module
 **   This class depends upon the existence of a python module "psdaq.configdb.<detType>_config.py", which contains
 **   the following functions:
 **     <detType>_init    (str)        - returns a handle (PyObject) for interacting with the device configuration
 **     <detType>_connect (handle)     - returns a dictionary with "paddr" element for Xpm link ID.  Other entries allowed.
 **     <detType>_config  (handle,...) - returns a json string containing configuration information to be stored in xtc.
 **     <detType>_unconfig(handle)     - return ignored
 **/

#include "drp.hh"
#include "Detector.hh"
#include "EventBatcher.hh"
#include "xtcdata/xtc/Xtc.hh"
#include "xtcdata/xtc/ConfigIter.hh"
#include "xtcdata/xtc/NamesId.hh"
#include <Python.h>

namespace Drp {
  class PythonConfigScanner;

  /** Segment limit used to space the names indices below. */
  enum { MaxSegsPerNode = 10  /**< Maximum number of detector segments per node (10). */ };
  /** NamesId index bases of BEB detectors, one block of MaxSegsPerNode indices each (the code comment says index for xtc NamesId). */
  enum {ConfigNamesIndex = NamesIndex::BASE, /**< Configuration names, starting at NamesIndex::BASE (0); BEBDetector::configure() uses ConfigNamesIndex + segment. */ 
        EventNamesIndex  = unsigned(ConfigNamesIndex) + unsigned(MaxSegsPerNode), /**< Event names, starting at 10. */ 
        UpdateNamesIndex = unsigned(EventNamesIndex)  + unsigned(MaxSegsPerNode)  /**< Scan update names, starting at 20; used by configureScan() and stepScan(). */ }; // index for xtc NamesId

/** Detector base for devices whose event data is formatted by the AxiStreamBatcherEventBuilder firmware (per the header comment). Configuration and control go through the Python module `psdaq.configdb.{detType}_config`, and each event is split into batcher subframes that are passed to the subclass _event(). */
class BEBDetector : public Detector
{
public:
    /** Construct the Detector base with virtual channel 1 and no Python module yet; subclasses must call _init() (or their own variant). */
    BEBDetector(Parameters* para, MemPool* pool);
    /** Delete the config scanner and release the Python module. */
    virtual ~BEBDetector();
public:  // Implementation of Detector
    /** Call `{detType}_connectionInfo` in Python with the device handle and msg, store its paddr entry (aborting if it is 0, 0xffffffff or has a low byte above 15), pass the result to _connectionInfo() and return xpmInfo(paddr). */
    nlohmann::json connectionInfo(const nlohmann::json& msg) override;
    /** Call `{detType}_connectionShutdown` in Python if the module defines it. */
    void           connectionShutdown() override;
    /** Keep the connect message and the readout group of this DRP (det_info.readout of entry collectionId). */
    void           connect       (const nlohmann::json&, const std::string& collectionId) override;
    /** Call `{detType}_config` in Python (with segment and serial numbers in multi-segment mode), translate the returned JSON, or list of JSON per segment, into configuration XTC, let the subclass _configure() add its names, append the configuration to xtc and return the _configure() result. Throws a C string, before translating, if the JSON text is longer than maxTrSize. Returns -1 (as unsigned) if translation fails. */
    unsigned       configure     (const std::string& config_alias, XtcData::Xtc& xtc, const void* bufEnd) override;
    // Avoid "overloaded virtual function "Drp::Detector::event" is only partially overridden" warning
    using Detector::event;
    /** Split the DMA buffer of each lane in event->mask into batcher subframes (one more batcher level is unwrapped when m_debatch is set), keep all subframes of the first lane that has more than 2 and subframe 2 of each later one, and pass them to _event(). */
    void           event         (XtcData::Dgram& dgram, const void* bufEnd, PGPEvent* event, uint64_t l1count) override;
    /** Call `{detType}_unconfig` in Python with the device handle. */
    void           shutdown      () override;

    /** Pass stepInfo to PythonConfigScanner::configure() with the update names ID and return its result. */
    unsigned configureScan(const nlohmann::json& stepInfo, XtcData::Xtc& xtc, const void* bufEnd) override;
    /** Pass stepInfo to PythonConfigScanner::step() with the update names ID and return its result. */
    unsigned stepScan     (const nlohmann::json& stepInfo, XtcData::Xtc& xtc, const void* bufEnd) override;

    /** Return the timing header of DMA buffer index: the data after the first batcher header, or after the second one when m_debatch is set. */
    Pds::TimingHeader* getTimingHeader(uint32_t index) const override;

    /** Return obj; if it is null, print the Python error, log it and abort. */
    static PyObject* _check(PyObject*);
protected:  // This is the sub class interface
    virtual void           _connectionInfo(PyObject*) {} // handle dictionary entries returned by <detType>_connect python
    virtual unsigned       _configure(XtcData::Xtc&, const void* bufEnd, XtcData::ConfigIter&)=0; // attach descriptions to xtc
    virtual void           _event    (XtcData::Xtc&,     // fill xtc from subframes
                                      const void* bufEnd,
                                      uint64_t l1count,
                                      std::vector< XtcData::Array<uint8_t> >&) {}
protected:
    void _init(const char*);  // Must call from subclass constructor
    void _init_feb();         // Must call from subclass constructor
    // Helper functions
    std::vector< XtcData::Array<uint8_t> > _subframes(void* buffer, unsigned length);
    std::vector< XtcData::Array<uint8_t> > _subframes(void* buffer, unsigned length, size_t nsubhint);
    static std::string _string_from_PyDict(PyObject*, const char* key);
protected:
    std::string          m_connect_json;  // info passed on connect phase
    unsigned             m_readoutGroup;  // readout group from connect
    PyObject*            m_module;        // python module
    PyObject*            m_root;          // handle for python functions
    unsigned             m_paddr;         // timing system link id
    PythonConfigScanner* m_configScanner;
    bool                 m_debatch;       // data is contained in an extra AxiStreamBatcherEventBuilder
    bool                 m_multiSegment = false;
    std::string          m_segNoStr{""};
  };

}
