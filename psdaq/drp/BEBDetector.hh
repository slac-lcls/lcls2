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

  enum { MaxSegsPerNode = 10 };
  enum {ConfigNamesIndex = NamesIndex::BASE,
        EventNamesIndex  = ConfigNamesIndex + MaxSegsPerNode,
        UpdateNamesIndex = EventNamesIndex  + MaxSegsPerNode }; // index for xtc NamesId

class BEBDetector : public Detector
{
public:
    BEBDetector(Parameters* para, MemPool* pool);
    virtual ~BEBDetector();
public:  // Implementation of Detector
    nlohmann::json connectionInfo(const nlohmann::json& msg) override;
    void           connect       (const nlohmann::json&, const std::string& collectionId) override;
    unsigned       configure     (const std::string& config_alias, XtcData::Xtc& xtc, const void* bufEnd) override;
    void           event         (XtcData::Dgram& dgram, const void* bufEnd, PGPEvent* event) override;
    void           shutdown      () override;

    unsigned configureScan(const nlohmann::json& stepInfo, XtcData::Xtc& xtc, const void* bufEnd) override;
    unsigned stepScan     (const nlohmann::json& stepInfo, XtcData::Xtc& xtc, const void* bufEnd) override;

    Pds::TimingHeader* getTimingHeader(uint32_t index) const override;

    static PyObject* _check(PyObject*);
protected:  // This is the sub class interface
    virtual void           _connectionInfo(PyObject*) {} // handle dictionary entries returned by <detType>_connect python
    virtual unsigned       _configure(XtcData::Xtc&, const void* bufEnd, XtcData::ConfigIter&)=0; // attach descriptions to xtc
    virtual void           _event    (XtcData::Xtc&,     // fill xtc from subframes
                                      const void* bufEnd,
                                      std::vector< XtcData::Array<uint8_t> >&) {}
protected:
    void _init(const char*);  // Must call from subclass constructor
    void _init_feb();         // Must call from subclass constructor
    // Helper functions
    std::vector< XtcData::Array<uint8_t> > _subframes(void* buffer, unsigned length);
    static std::string _string_from_PyDict(PyObject*, const char* key);
protected:
    std::string          m_connect_json;  // info passed on connect phase
    unsigned             m_readoutGroup;  // readout group from connect
    PyObject*            m_module;        // python module
    PyObject*            m_root;          // handle for python functions
    unsigned             m_paddr;         // timing system link id
    PythonConfigScanner* m_configScanner;
    bool                 m_debatch;       // data is contained in an extra AxiStreamBatcherEventBuilder
  };

}
